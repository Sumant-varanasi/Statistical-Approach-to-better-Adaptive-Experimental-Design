"""
The smart (physics-informed) reactor advisor, and the baselines it is compared with.

Plain Bayesian optimisation (bo/) treats the reactor as a black box: it only sees one number
per experiment (the profit) and learns the landscape from scratch.  The advisor instead
knows the PHYSICS of the reactor and learns only what it does not know:

  KNOWN (built in)                         LEARNED from the experiments
  ---------------------------------------  ---------------------------------------------
  which reactions happen, stoichiometry    rate constants k_ref of every reaction
  mole balances over the catalyst surface  activation energies Ea (temperature response)
  Arrhenius temperature dependence         product adsorption (inhibition) and its heat
  ideal-gas partial pressures              catalyst activity loss with time-on-stream
  equilibrium limit from thermodynamics    a correction for anything the physics misses
  the economics (prices)                   (Gaussian process on the profit residual)

Every experiment gives the full GC composition (a mole fraction per species), so one
experiment teaches the model about EVERY reaction at once, not just "profit was 12.3".

How a campaign goes (same for every method, because that is how the reactor is run):
  * the catalyst is loaded once (WCAT) and never changed
  * the operator's 5 screening runs come first (systems.screening_runs)
  * from then on every suggested run must lie within MAX_STEP of the previous run for every
    knob, inside the safety box, and leave enough total flow for the GC
  * optionally a check run (the base case repeated) every few runs, to see drift

How the advisor chooses the next run (v5, planner="horizon"):
  1. fit the kinetics to all GC readings so far (kinetics.py), with its uncertainty, choosing
     the model class (inhibition, its temperature dependence, deactivation) the data support
  2. draw a handful of plausible versions of the chemistry consistent with the data
  3. PLAN: find where the model expects the best profit over the whole box (x*)
  4. for the candidate first steps within reach, roll the model forward H runs (greedy on the
     model mean, one allowed step at a time) and score the whole trajectory under every
     plausible chemistry:  return = sum of the H profits + (R - H) x the best profit found,
     R = runs the operator can still afford.  So a step that earns nothing now but opens a
     better region is chosen when the campaign is long enough to cash it in, and a safe
     step is chosen when it is not.
  5. a Gaussian-process correction, fitted to what the physics got wrong so far, is added to
     every prediction (with a noise floor from the GC error), and the predictive spread is
     calibrated against the advisor's own past misses
  6. product specs (minimum selectivity, maximum adiabatic temperature rise, ...) are
     enforced by the probability, over the plausible chemistries, that a run meets them
planner="greedy" keeps the v4 rule (Expected Improvement one step ahead, with a purposeful
step toward x* when nothing within reach beats the best run), for comparison.
"""
import json
import os
import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution, minimize
from .systems import (SYSTEMS, MODELS, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, WCAT, A_SPEC, DIM, R, T_REF,
                      unscale, scale, physical, feasible, step_scaled, base_case, screening_runs)
from .kinetics import KineticFit, GC_NOISE_REL, GC_NOISE_ABS, HOURS_PER_RUN
from bo.gp import GP
from bo.loop import LogInputs, copula_transform
from bo.acquisition import expected_improvement


# =================================================================================== the virtual reactor
class VirtualReactor:
    """Stands in for the real reactor + GC.  Uses the TRUE kinetics (with the adsorption terms
    the advisor's model does not fully have) and adds GC noise:
        y_GC = y_true * (1 + 1 % noise) + 0.05 % absolute noise, renormalised
    (typical repeatability of an online GC with an internal standard).
    The operator never sees profit directly: it is computed from the GC reading using the
    inert as an internal standard, exactly as the advisor tool does with real data.

    deactivation: base rate of activity loss per hour on stream (0 = a stable catalyst).  The
    true loss is severity-weighted (faster when hotter and with more A in the feed, as coking
    is), a form the advisor's model does not know: it assumes a constant rate."""

    def __init__(self, system, seed=0, wcat=WCAT, a_spec=A_SPEC, gc_rel=GC_NOISE_REL, gc_abs=GC_NOISE_ABS,
                 deactivation=0.0, hours_per_run=HOURS_PER_RUN):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.rng = np.random.default_rng(50_000 + seed)
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.gc_rel, self.gc_abs = gc_rel, gc_abs
        self.deactivation, self.hours_per_run = float(deactivation), float(hours_per_run)
        self.ln_activity, self.t_on_stream, self.n_runs = 0.0, 0.0, 0

    def activity(self):
        return float(np.exp(self.ln_activity))

    def loss_rate(self, x):
        """True activity loss per hour at settings x: Arrhenius in T (E = 40 kJ/mol) and
        growing with the partial pressure of A (coke precursor), 1x at 10 bar with 40 % A."""
        P, VA, VB, VI, T = unscale(x)[0]
        PA = P * VA / (VA + VB + VI)
        return self.deactivation * np.exp(-40e3 / R * (1 / T - 1 / T_REF)) * np.sqrt(max(PA, 0.0) / 4.0)

    def true_flows(self, x, ln_act=None):
        P, VA, VB, VI, T = unscale(x)[0]
        act = np.exp(self.ln_activity if ln_act is None else ln_act)
        F, F0 = self.S.simulate(P, VA, VB, VI, self.wcat, T, truth=True, method="lsoda", a_spec=self.a_spec * act)
        return F, F0, P

    def true_profit(self, x, ln_act=None):
        """True profit of settings x at the current activity (or at ln_act)."""
        F, F0, P = self.true_flows(x, ln_act)
        return float(self.S.profit(F, F0, [P], [self.wcat])[0])

    def run(self, x, hours=None):
        hours = self.hours_per_run if hours is None else float(hours)
        F, F0, P = self.true_flows(x)
        y = F / F.sum(1, keepdims=True)
        y = y * (1 + self.gc_rel * self.rng.standard_normal(y.shape)) \
            + self.gc_abs * self.rng.standard_normal(y.shape)
        y = np.maximum(y, 0.0)
        y /= y.sum(1, keepdims=True)
        ph6 = physical(x, self.wcat)
        XA, SC, _ = kpis_from_gc(self.S, ph6, y)
        out = dict(gc=y[0], profit=float(profit_from_gc(self.S, ph6, y)[0]),
                   profit_true=float(self.S.profit(F, F0, [P], [self.wcat])[0]),
                   X_A=float(XA[0]), S_C=float(SC[0]), hours=hours, activity=self.activity(),
                   dT_ad=float(self.S.dT_ad(F, F0)[0]))
        self.ln_activity -= self.loss_rate(x) * hours     # the catalyst ages during the run
        self.t_on_stream += hours
        self.n_runs += 1
        return out


def profit_from_gc(S, phys6, y):
    """Profit from settings + a GC reading (inert = internal standard).  This is all the
    operator has to provide: what they set, and what the GC printed.
    phys6 rows: P, V_A, V_B, V_I, T, Wcat."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    F = S.flows_from_gc(y, VA, VB, VI)
    F0 = S.inlet_flows(VA, VB, VI)
    return S.profit(F, F0, P, W)


def kpis_from_gc(S, phys6, y):
    """Conversion, selectivity and yield as the operator would read them off the GC."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    return S.kpis(S.flows_from_gc(y, VA, VB, VI), S.inlet_flows(VA, VB, VI))


def dT_ad_from_gc(S, phys6, y):
    """Adiabatic temperature rise implied by a GC reading (what the isothermal model cannot see)."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    return S.dT_ad(S.flows_from_gc(y, VA, VB, VI), S.inlet_flows(VA, VB, VI))


def profit_noise_sd(S, phys6, y, n=64, seed=0):
    """How much GC noise alone moves the profit computed from a reading (Monte Carlo with
    the GC error model).  Profit is a small difference of large numbers, so 1 % GC noise
    becomes several money units per hour; the correction GP must not mistake that for a
    real effect."""
    rng = np.random.default_rng(seed)
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        Yn = y[i] * (1 + GC_NOISE_REL * rng.standard_normal((n, y.shape[1]))) \
            + GC_NOISE_ABS * rng.standard_normal((n, y.shape[1]))
        Yn = np.maximum(Yn, 0)
        Yn /= Yn.sum(1, keepdims=True)
        out[i] = profit_from_gc(S, np.repeat(phys6[i:i + 1], n, 0), Yn).std()
    return out


class SmoothGP(GP):
    """GP for the physics correction: length-scales of at least 0.15 (in scaled units).
    What the physics gets wrong (here: product inhibition) changes slowly across the box;
    anything wigglier than that is measurement noise, not chemistry."""
    LS_BOUNDS = (np.log(0.15), np.log(20.0))


# =================================================================================== step limits
def step_box(x_prev, step, lo, hi, fixed=None):
    """Scaled box of runs reachable from x_prev in one step, inside the safety box."""
    blo = np.maximum(lo, x_prev - step)
    bhi = np.minimum(hi, x_prev + step)
    for i, v in (fixed or {}).items():
        blo[i] = bhi[i] = v
    return blo, np.maximum(bhi, blo)


def sobol_in(lo, hi, n, rng, free=None):
    """n Sobol points in the box; with `free` (indices), only those dimensions vary."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if free is None or len(free) == DIM:
        return lo + qmc.Sobol(DIM, scramble=True, seed=int(rng.integers(1 << 31))).random(n) * (hi - lo)
    out = np.tile(lo, (n, 1))
    if len(free):
        out[:, free] = lo[free] + qmc.Sobol(len(free), scramble=True, seed=int(rng.integers(1 << 31))).random(n) * (hi - lo)[free]
    return out


def keep_feasible(C):
    return C[feasible(unscale(C))]


# =================================================================================== the advisor
SPEC_KEYS = {"S_C": ("min", "selectivity"), "X_A": ("min", "conversion"), "dT_ad": ("max", "adiabatic temperature rise")}


class SmartAdvisor:
    """
    adv = SmartAdvisor('complex', wcat=1.0, a_spec=100)
    adv.update(X, gc, profit, hours=...)   X scaled (n x 5), gc (n x ns), measured profit (n,)
    x, info = adv.suggest(x_prev)          next run (scaled), reachable from x_prev, and why
    adv.predict(Xc)                        predicted profit mean and sd at candidates
    adv.recommend()                        index of the best tested run (by prediction, meeting the spec)
    adv.plan()                             where the model expects the best profit (x*, mean, sd)

    wcat, a_spec  catalyst mass (g) and BET area (m2/g) of the loaded bed
    max_step      largest change per knob between consecutive runs (physical units)
    lo, hi        optional safety box in scaled units (e.g. never above 700 K)
    fixed         {knob index: scaled value} knobs the operator cannot change
    spec          product / safety spec, e.g. {"S_C": 0.9, "dT_ad": 60}: minimum selectivity,
                  maximum adiabatic temperature rise (K); also "X_A" (minimum conversion)
    planner       "horizon" (v5 lookahead) or "greedy" (v4 one-step rule)
    horizon       runs the lookahead rolls forward (H)
    remaining     runs the operator can still afford (R); the planner weighs exploration by it
    residual_gp   correct the physics with a GP on the profit residual (on by default)
    calibrate     scale the predictive spread by the advisor's own past misses
    models        kinetic model classes the fit may choose from (kinetics.MODELS); None = all
    """

    def __init__(self, system, wcat=WCAT, a_spec=A_SPEC, seed=0, n_theta=6, lo=None, hi=None,
                 fixed=None, max_step=None, residual_gp=True, spec=None, planner="horizon", horizon=4,
                 remaining=10, calibrate=True, models=None):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.kin = KineticFit(self.S, a_spec=self.a_spec, seed=seed, **({} if models is None else dict(models=models)))
        self.n_theta = n_theta
        self.rng = np.random.default_rng(seed)
        self.lo = np.zeros(DIM) if lo is None else np.asarray(lo, float)
        self.hi = np.ones(DIM) if hi is None else np.asarray(hi, float)
        self.fixed = dict(fixed or {})
        self.free = [i for i in range(DIM) if i not in self.fixed]
        self.step = step_scaled(max_step)
        self.use_gp = residual_gp
        self.spec = {k: float(v) for k, v in (spec or {}).items() if v is not None and k in SPEC_KEYS}
        self.planner, self.horizon, self.remaining = planner, int(horizon), int(remaining)
        self.calibrate = calibrate
        self.calib = 1.0
        self.gp = None
        self.seed = seed
        self.n = 0
        self.t_next = 0.0

    # ------------------------------------------------------------------ learning
    def update(self, X, gc, profit, wcat_rows=None, hours=None):
        """wcat_rows: catalyst mass of each logged run (defaults to the campaign's wcat);
        hours: duration of each run (time-on-stream clock; default 24 h each)."""
        self.X = np.atleast_2d(np.asarray(X, float))
        self.gc = np.atleast_2d(gc)
        self.y = np.asarray(profit, float)
        n = len(self.y)
        self.n = n
        self.W = np.broadcast_to(np.asarray(self.wcat if wcat_rows is None else wcat_rows, float), (n,)).copy()
        self.hours = np.full(n, HOURS_PER_RUN) if hours is None else np.asarray(hours, float)
        ph6 = physical(self.X, self.W)
        n_starts = None if self.kin.n_fits == 0 else (3 if n % 5 == 0 else 1)
        self.kin.fit(ph6, self.gc, hours=self.hours, n_starts=n_starts)
        self.t_rows = self.kin.t_rows
        self.t_next = float(self.t_rows[-1] + self.hours[-1])
        self.thetas = self.kin.samples(self.n_theta, self.rng)
        self.phys_at_data = self._physics(self.X, self.kin.theta[None], W=self.W, t=self.t_rows)[0]
        self.resid = self.y - self.phys_at_data
        self.gp = None
        if self.use_gp and n >= 4:
            # the GP's noise is not allowed to drop below what the GC noise implies
            s = max(self.resid.std(), 1e-6)
            nf = np.mean(profit_noise_sd(self.S, ph6, self.gc) ** 2) / s ** 2
            gp = SmoothGP(seed=self.seed, noise_bounds=(min(nf, 0.5), max(1.0, 2 * nf)))
            self.gp = LogInputs(gp, (0,), LOW, HIGH).fit(self.X, self.resid)     # GP sees log P
        # calibration: z-scores of the measured profits against the predictive spread of an
        # observation at the data (physics disagreement + GP, noise included); inflate if > 1
        self.calib = 1.0
        if self.calibrate and n >= 6:
            mu, sd, _ = self._predict(self.X, t=self.t_rows, include_noise=True)
            z = (self.y - mu) / np.maximum(sd, 1e-9)
            self.calib = float(np.clip(np.sqrt(np.mean(z ** 2)), 1.0, 5.0))
        self.mu_data = self.predict(self.X, t=self.t_rows)[0]
        return self

    def _physics(self, Xc, thetas, W=None, kpis=False, t=None):
        """Profit predicted by the physics model for every (theta, candidate) pair: (m, n).
        With kpis=True also returns conversion, selectivity and adiabatic rise, (m, n) each.
        t = time-on-stream of the candidates (default: the next run's)."""
        m, n = len(thetas), len(Xc)
        ph = np.tile(unscale(Xc), (m, 1))
        P = ph[:, 0]
        Wr = np.tile(np.broadcast_to(np.asarray(self.wcat if W is None else W, float), (n,)), m)
        tt = np.tile(np.broadcast_to(np.asarray(self.t_next if t is None else t, float), (n,)), m)
        F, F0 = self.kin.simulate(np.column_stack([ph, Wr]), np.repeat(thetas, n, 0), t=tt)
        with np.errstate(invalid="ignore"):
            pr = self.S.profit(F, F0, P, Wr).reshape(m, n)
        pr[np.isinf(pr)] = np.nan                      # NaN = this chemistry gave no answer here
        if not kpis:
            return pr
        with np.errstate(invalid="ignore"):
            XA, SC, _ = self.S.kpis(F, F0)
            dT = self.S.dT_ad(F, F0)
        return pr, XA.reshape(m, n), SC.reshape(m, n), dT.reshape(m, n)

    def _predict(self, Xc, t=None, include_noise=False, thetas=None):
        Xc = np.atleast_2d(Xc)
        pr = self._physics(Xc, self.thetas if thetas is None else thetas, t=t)
        mu_p = np.nan_to_num(pr[0], nan=-1e3)          # best-estimate chemistry
        with np.errstate(invalid="ignore"):
            sd_p = np.nan_to_num(np.nanstd(pr, 0), nan=0.0)   # disagreement between plausible chemistries
        if self.gp is not None:
            mu_r, sd_r = self.gp.predict(Xc, include_noise=include_noise)
        else:
            mu_r, sd_r = np.zeros(len(Xc)), np.zeros(len(Xc))
        return mu_p + mu_r, np.sqrt(sd_p ** 2 + sd_r ** 2 + 1e-9), dict(mu_phys=mu_p, sd_phys=sd_p, mu_corr=mu_r, sd_corr=sd_r)

    def predict(self, Xc, parts=False, t=None):
        mu, sd, p = self._predict(Xc, t=t)
        sd = sd * self.calib
        return (mu, sd, p) if parts else (mu, sd)

    def feasibility(self, Xc, t=None):
        """P(meets the spec) over the plausible chemistries, and the mean KPIs: (n,), dict."""
        Xc = np.atleast_2d(Xc)
        pr, XA, SC, dT = self._physics(Xc, self.thetas, kpis=True, t=t)
        ok = np.ones_like(pr, bool)
        for key, lim in self.spec.items():
            val = {"S_C": SC, "X_A": XA, "dT_ad": dT}[key]
            ok &= (val >= lim) if SPEC_KEYS[key][0] == "min" else (val <= lim)
        ok &= np.isfinite(pr)
        with np.errstate(invalid="ignore"):
            kp = dict(X_A=np.nanmean(XA, 0), S_C=np.nanmean(SC, 0), dT_ad=np.nanmean(dT, 0))
        return ok.mean(0), kp

    def recommend(self):
        """Index of the tested run with the best PREDICTED profit (a single lucky GC reading
        cannot fool it) among the runs that met the spec, measured; falls back to all runs."""
        mu = self.mu_data
        if self.spec:
            ph6 = physical(self.X, self.W)
            XA, SC, _ = kpis_from_gc(self.S, ph6, self.gc)
            dT = dT_ad_from_gc(self.S, ph6, self.gc)
            ok = np.ones(self.n, bool)
            for key, lim in self.spec.items():
                val = {"S_C": SC, "X_A": XA, "dT_ad": dT}[key]
                ok &= (val >= lim) if SPEC_KEYS[key][0] == "min" else (val <= lim)
            if ok.any():
                idx = np.nonzero(ok)[0]
                return int(idx[np.argmax(mu[idx])])
        return int(np.argmax(mu))

    # ------------------------------------------------------------------ planning
    def _pin(self, C):
        C = np.clip(C, self.lo, self.hi)
        for i, v in self.fixed.items():
            C[:, i] = v
        return C

    def _ok(self, C):
        """Feasible (flow minimum) and, with a spec, likely to meet it."""
        keep = feasible(unscale(C))
        if self.spec and keep.any():
            pf = np.zeros(len(C))
            pf[keep] = self.feasibility(C[keep])[0]
            keep &= pf >= 0.5
        return keep

    def plan(self, rng=None):
        """Where the model expects the best profit over the WHOLE safety box (not just one
        step away): x*, its predicted profit and uncertainty."""
        rng = rng or self.rng
        x_best = self.X[int(np.argmax(self.mu_data))]
        C1 = np.vstack([sobol_in(self.lo, self.hi, 1024, rng, self.free), self.X,
                        x_best + rng.normal(0, 0.5, (64, DIM)) * self.step])
        C1 = self._pin(C1)
        keep = self._ok(C1)
        if not keep.any():                      # nothing meets the spec anywhere: plan without it
            keep = feasible(unscale(C1))
            if not keep.any():
                return x_best.copy(), float(self.mu_data.max()), 0.0
        C1 = C1[keep]
        mu1 = self.predict(C1)[0]
        top = C1[np.argsort(mu1)[-5:]]
        C2 = self._pin(top[rng.integers(len(top), size=256)] + rng.normal(0, 0.03, (256, DIM)))
        C2 = C2[self._ok(C2)]
        mu2, sd2 = self.predict(C2) if len(C2) else (np.array([]), np.array([]))
        C, mu, sd = np.vstack([C1, C2]), np.r_[mu1, mu2], np.r_[np.full(len(C1), np.nan), sd2]
        k = int(np.argmax(mu))
        if np.isnan(sd[k]):
            sd[k] = self.predict(C[k:k + 1])[1][0]
        return C[k], float(mu[k]), float(sd[k])

    def steps_away(self, x_from, x_to):
        """How many runs it takes to get from x_from to x_to under the step limits."""
        d = np.abs(np.asarray(x_to) - np.asarray(x_from))
        return int(np.ceil(np.max(d / np.where(self.step > 0, self.step, 1.0) - 1e-9)))

    def _step_candidates(self, x_from, x_star, rng, n_sobol):
        """Structured candidates for one allowed step from x_from: the step toward x*, the box
        corners (full steps on every free knob), points on the line to x*, stay, Sobol."""
        blo, bhi = step_box(x_from, self.step, self.lo, self.hi, self.fixed)
        pts = [np.clip(x_star, blo, bhi), x_from.copy()]
        for f in (0.25, 0.5, 0.75):
            pts.append(np.clip(x_from + f * (x_star - x_from), blo, bhi))
        nf = len(self.free)
        if nf:
            corners = np.array(np.meshgrid(*[[0, 1]] * nf)).reshape(nf, -1).T
            for c in corners:
                p = x_from.copy()
                p[self.free] = np.where(c == 1, bhi[self.free], blo[self.free])
                pts.append(p)
        C = np.vstack([np.array(pts), sobol_in(blo, bhi, n_sobol, rng, self.free)])
        return self._pin(C)

    def suggest(self, x_prev=None, rng=None):
        """The next run, reachable from x_prev (default: the last logged run) in one step."""
        rng = rng or self.rng
        x_prev = np.asarray(self.X[-1] if x_prev is None else x_prev, float)
        if self.planner == "greedy":
            return self._suggest_greedy(x_prev, rng)
        x_star, mu_star, sd_star = self.plan(rng)
        best = float(self.mu_data.max())
        H = max(1, min(self.horizon, self.remaining))
        R = max(self.remaining, H)
        # ---- first-step candidates
        C1 = self._step_candidates(x_prev, x_star, rng, 256)
        keep = self._ok(C1)
        if not keep.any():
            # nothing within one step is likely to meet the spec (or the flow minimum): take the
            # step that gets closest to meeting it, heading toward the model optimum
            flow_ok = feasible(unscale(C1))
            if not flow_ok.any():
                return x_prev.copy(), dict(pred=best, sd=0.0, ei=0.0, best_pred=best, model_opt=x_star,
                                           model_opt_pred=mu_star, model_opt_sd=sd_star,
                                           steps_away=self.steps_away(x_prev, x_star), toward_opt=False,
                                           mode="stay", score=best, p_spec=0.0)
            Cf = C1[flow_ok]
            pf = self.feasibility(Cf)[0] if self.spec else np.ones(len(Cf))
            dist = np.max(np.abs(Cf - x_star) / np.where(self.step > 0, self.step, 1.0), axis=1)
            j = int(np.lexsort((dist, -np.round(pf, 2)))[0])            # most likely to meet the spec, then closest to x*
            x = Cf[j]
            mu, sd = self.predict(x[None])
            return x, dict(pred=float(mu[0]), sd=float(sd[0]), ei=0.0, best_pred=best, model_opt=x_star,
                           model_opt_pred=mu_star, model_opt_sd=sd_star,
                           steps_away=self.steps_away(x_prev, x_star), toward_opt=True,
                           mode="recover", score=float(mu[0]), p_spec=float(pf[j]))
        C1 = C1[keep]
        mu1, sd1 = self.predict(C1)
        ei1 = expected_improvement(mu1, sd1, best)
        pick = set(np.argsort(ei1)[-6:]) | set(np.argsort(mu1)[-3:]) | {0, 1}   # 0 = toward x*, 1 = stay
        M = sorted(pick)
        # dedupe within a quarter step
        chosen = []
        for i in M:
            if all(np.max(np.abs(C1[i] - C1[j]) / np.where(self.step > 0, self.step, 1)) > 0.25 for j in chosen):
                chosen.append(i)
        M = chosen
        # ---- greedy rollout on the model mean, batched per depth over all first steps
        traj = [np.array([C1[i] for i in M])]                                 # (m, DIM) per depth
        for h in range(1, H):
            cand = [self._step_candidates(x, x_star, rng, 24) for x in traj[-1]]
            sizes = [len(c) for c in cand]
            allc = np.vstack(cand)
            ok = self._ok(allc)
            mu = self._predict(allc, thetas=self.thetas[:1])[0]
            mu[~ok] = -np.inf
            nxt, start = [], 0
            for x, sz in zip(traj[-1], sizes):
                seg = mu[start:start + sz]
                nxt.append(allc[start + int(np.argmax(seg))] if np.isfinite(seg.max()) else x)
                start += sz
            traj.append(np.array(nxt))
        T = np.stack(traj, 1)                                                   # (m, H, DIM)
        # ---- score every trajectory under every plausible chemistry (+ GP draws)
        flat = T.reshape(-1, DIM)
        pr = self._physics(flat, self.thetas)                                   # (n_theta, m*H)
        pr = np.nan_to_num(pr, nan=-1e3)
        mu_r, sd_r = (self.gp.predict(flat) if self.gp is not None else (np.zeros(len(flat)), np.zeros(len(flat))))
        z = rng.standard_normal((self.n_theta, 1))
        f = pr + mu_r[None, :] + z * (sd_r * self.calib)[None, :]
        f = f.reshape(self.n_theta, len(M), H)
        ret = f.sum(2) + (R - H) * np.maximum(best, f.max(2))                   # (n_theta, m)
        score = ret.mean(0) / R                                                 # average profit per remaining run
        j = int(np.argmax(score))
        x = C1[M[j]]
        i1 = M[j]
        d_opt, d_step = x_star - x_prev, x - x_prev
        toward = bool(np.dot(d_opt, d_step) > 0.5 * np.linalg.norm(d_opt) * np.linalg.norm(d_step) + 1e-12)
        if np.max(np.abs(x - x_prev)) < 1e-9:
            mode = "stay"
        elif mu1[i1] - best > 1.64 * sd1[i1]:
            mode = "improve"
        elif toward and mu1[i1] < best:
            mode = "walk"
        else:
            mode = "check"
        pf, kp = (self.feasibility(x[None]) if self.spec else (np.array([1.0]), {}))
        return x, dict(pred=float(mu1[i1]), sd=float(sd1[i1]), ei=float(ei1[i1]), best_pred=best,
                       model_opt=x_star, model_opt_pred=mu_star, model_opt_sd=sd_star,
                       steps_away=self.steps_away(x_prev, x_star), toward_opt=toward, mode=mode,
                       score=float(score[j]), p_spec=float(pf[0]), trajectory=T[j],
                       traj_profit=[float(v) for v in f[0, j]])

    def _suggest_greedy(self, x_prev, rng):
        """The v4 rule: Expected Improvement one step ahead, and a purposeful step on the
        bottleneck knob toward x* when nothing within reach beats the best run."""
        x_star, mu_star, sd_star = self.plan(rng)
        blo, bhi = step_box(x_prev, self.step, self.lo, self.hi, self.fixed)
        waypoint = np.clip(x_star, blo, bhi)
        best = float(self.mu_data.max())
        x_best = self.X[int(np.argmax(self.mu_data))]
        C = np.vstack([sobol_in(blo, bhi, 512, rng, self.free), waypoint[None],
                       np.clip(waypoint + rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi),
                       np.clip(x_best + rng.normal(0, 0.3, (64, DIM)) * self.step, blo, bhi)])
        C = self._pin(C)
        ok = self._ok(C)
        if not ok.any():
            return x_prev.copy(), dict(pred=best, sd=0.0, ei=0.0, best_pred=best, model_opt=x_star,
                                       model_opt_pred=mu_star, model_opt_sd=sd_star,
                                       steps_away=self.steps_away(x_prev, x_star), toward_opt=False, mode="stay")
        C = C[ok]
        mu, sd = self.predict(C)
        ei = expected_improvement(mu, sd, best)
        j = int(np.argmax(ei))
        mode = "improve"
        far_gain = mu_star - 1.64 * sd_star - best
        if far_gain > 0 and ei[j] < 0.1 * far_gain:
            dist = np.abs(x_star - x_prev) / np.where(self.step > 0, self.step, 1.0)
            k = int(np.argmax(dist))
            progress = np.abs(x_star[k] - C[:, k]) <= np.abs(x_star[k] - x_prev[k]) - 0.9 * self.step[k]
            if progress.any():
                cand = np.nonzero(progress)[0]
                j = int(cand[np.argmax(mu[cand])])
            else:
                j = int(np.argmin(np.max(np.abs(C - waypoint), axis=1)))
            mode = "walk"
        x = C[j]
        d_opt, d_step = x_star - x_prev, x - x_prev
        toward = bool(np.dot(d_opt, d_step) > 0.5 * np.linalg.norm(d_opt) * np.linalg.norm(d_step) + 1e-12)
        return x, dict(pred=float(mu[j]), sd=float(sd[j]), ei=float(ei[j]), best_pred=best,
                       model_opt=x_star, model_opt_pred=mu_star, model_opt_sd=sd_star,
                       steps_away=self.steps_away(x_prev, x_star), toward_opt=toward,
                       waypoint=waypoint, mode=mode)


# =================================================================================== campaigns
def _new_history():
    return dict(X=[], gc=[], y_obs=[], y_true=[], X_A=[], S_C=[], dT_ad=[], hours=[], activity=[], rec=[], check=[])


def _evaluate(vr, h, x, check=False):
    r = vr.run(x)
    h["X"].append(np.asarray(x, float).copy()); h["gc"].append(r["gc"])
    h["y_obs"].append(r["profit"]); h["y_true"].append(r["profit_true"])
    h["X_A"].append(r["X_A"]); h["S_C"].append(r["S_C"]); h["dT_ad"].append(r["dT_ad"])
    h["hours"].append(r["hours"]); h["activity"].append(r["activity"]); h["check"].append(bool(check))
    h["rec"].append(h["y_true"][int(np.argmax(h["y_obs"]))])
    return r


def _score_rec(vr, h, idx):
    """True profit, at the CURRENT catalyst activity, of the settings the method recommends."""
    h["rec"][-1] = vr.true_profit(h["X"][idx])


def start_campaign(system, seed=0, wcat=WCAT, a_spec=A_SPEC, deactivation=0.0):
    """The virtual reactor and the operator's 5 screening runs (shared by every method)."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed, wcat, a_spec, deactivation=deactivation)
    h = _new_history()
    base = base_case(seed)
    for row in screening_runs(base):
        _evaluate(vr, h, scale(row)[0])
    return S, vr, h, base


def is_check_run(n, check_every):
    """Protocol: after the 5 screening runs, repeat the base case every `check_every` runs."""
    return bool(check_every) and n >= 5 and (n - 5) % check_every == check_every - 1


def _pack(h, **extra):
    out = {k: np.array(v) for k, v in h.items()}
    out["cum_true"] = np.cumsum(out["y_true"])
    out.update(extra)
    return out


def run_smart(system, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, verbose=False, model=None, callback=None,
              planner="horizon", models=None, deactivation=0.0, check_every=0, spec=None):
    """One campaign with the smart advisor on the virtual reactor.
    model: name in systems.MODELS to give the advisor a deliberately WRONG reaction network
    (stress test); default = the right network (still without the reactant adsorption).
    planner / models: "horizon" + all kinetic models = v5; "greedy" + ("plain",) = v4.
    callback(run_number, x, result, info) is called after every advisor-chosen run."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec, deactivation)
    x_base = scale(base)[0]
    adv = SmartAdvisor(MODELS[model] if model else S, wcat=wcat, a_spec=a_spec, seed=seed, planner=planner,
                       models=models, spec=spec, remaining=budget - 5)
    rng = np.random.default_rng(seed)
    info_log = []
    while len(h["X"]) < budget:
        adv.remaining = budget - len(h["X"])
        adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]), hours=np.array(h["hours"]))
        _score_rec(vr, h, adv.recommend())
        n = len(h["X"])
        if is_check_run(n, check_every):
            x, info, chk = x_base.copy(), dict(mode="check-run", pred=float("nan"), sd=0.0), True
        else:
            x, info = adv.suggest(h["X"][-1], rng)
            chk = False
        r = _evaluate(vr, h, x, check=chk)
        info_log.append(info)
        if len(h["X"]) == budget:
            adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]), hours=np.array(h["hours"]))
        _score_rec(vr, h, adv.recommend())
        if callback:
            callback(len(h["X"]), x, r, info)
        if verbose:
            print(f"smart {system} run {len(h['X']):2d}: measured {r['profit']:7.2f}  predicted "
                  f"{info.get('pred', float('nan')):7.2f} +- {info.get('sd', 0):.2f}   recommended (true) {h['rec'][-1]:7.2f}")
    name = "smart" if planner == "horizon" and models is None else "smart_v4"
    if model:
        name += f"_{model}"
    return _pack(h, strategy=name, system=system, seed=seed, base=base, theta=adv.kin.theta,
                 kin_model=adv.kin.model, info=info_log, advisor=adv)


# ----------------------------------------------------------------------------------- baselines
class ChemistPolicy:
    """What an experienced operator does by hand: read conversion and selectivity off the GC,
    apply rules of thumb, and keep whatever worked.

      selectivity poor (< 0.85): cooler (side reactions have the higher activation energy),
                                 compensating with more pressure; then more B; then dilute
      conversion poor  (< 0.70): hotter and more pressure; then less flow (longer contact);
                                 then less inert
      both fine:                 push throughput (more A and B, with pressure); then hotter;
                                 then less inert

    Momentum: if the last change improved on the best run, repeat it.  Otherwise go back to
    the best run and try the next rule.  When a new best appears the rule list restarts.
    A move is one full step of each knob involved, clipped to the step limits from the
    previous run, and a chemist never repeats an experiment: a move that lands on a point
    already run is skipped.  When every rule has been tried from the current best, the
    operator takes a random allowed step ("try something").  Uses only what a person can
    see: the GC reading and the profit."""
    S_LOW, X_LOW = 0.85, 0.70
    KNOB = {"T": 4, "P": 0, "A": 1, "B": 2, "I": 3}

    def __init__(self, step, lo, hi, fixed=None, seed=0):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 31)
        self.last, self.best_idx, self.k = None, -1, 0

    def rules(self, XA, SC):
        if np.isnan(SC) or SC < self.S_LOW:
            return ["T- P+", "T-", "P+", "B+", "I+", "T- P+ B+"]
        if XA < self.X_LOW:
            return ["T+ P+", "P+", "T+", "A- B-", "I-", "T+ P+ I-"]
        return ["A+ B+", "A+ B+ P+", "P+", "T+", "I-", "A+ B+ T+"]

    def _apply(self, origin, name, x_last, X):
        x = origin.copy()
        for tok in name.split():
            i, d = self.KNOB[tok[0]], (+1 if tok[1] == "+" else -1)
            if i in self.fixed:
                return None
            x[i] = origin[i] + d * self.step[i]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        x = np.clip(x, blo, bhi)
        if not feasible(unscale(x))[0]:
            return None
        if any(np.max(np.abs(x - xr)) < 1e-3 for xr in X):          # already run: pointless
            return None
        return x

    def next(self, h):
        X, Y = h["X"], h["y_obs"]
        ib = int(np.argmax(Y))
        x_last = X[-1]
        if ib == len(X) - 1 and self.last is not None:            # last change worked: do it again
            x = self._apply(x_last, self.last, x_last, X)
            if x is not None:
                return x
        if ib != self.best_idx:
            self.best_idx, self.k = ib, 0
        lst = self.rules(h["X_A"][ib], h["S_C"][ib])
        for _ in range(len(lst)):
            name = lst[self.k % len(lst)]
            self.k += 1
            x = self._apply(X[ib], name, x_last, X)
            if x is not None:
                self.last = name
                return x
        self.last = None                                           # every rule tried: try something
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        for _ in range(50):
            x = blo + self.rng.random(DIM) * (bhi - blo)
            if feasible(unscale(x))[0]:
                return x
        return x_last.copy()

    def recommend(self, h):
        return int(np.argmax(h["y_obs"]))


class PlainBOPolicy:
    """Black-box Bayesian optimisation under the same step limits: the fitted GP of v2
    (Matern, ARD, log P, rank output transform) + Expected Improvement, but it sees only the
    profit numbers and may only propose runs within one step of the previous run."""

    def __init__(self, step, lo, hi, seed=0, fixed=None):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 11)
        self.seed = seed
        self.gp = None

    def _fit(self, h):
        X, Y = np.array(h["X"]), np.array(h["y_obs"])
        self.gp = LogInputs(GP(seed=self.seed), (0,), LOW, HIGH).fit(X, copula_transform(Y))
        self.X = X

    def next(self, h):
        self._fit(h)
        X = self.X
        x_last = X[-1]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        x_best = X[int(np.argmax(self.gp.predict(X)[0]))]
        C = np.vstack([sobol_in(blo, bhi, 512, self.rng),
                       np.clip(x_best + self.rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi)])
        for i, v in self.fixed.items():
            C[:, i] = v
        C = C[feasible(unscale(C))]
        if len(C) == 0:
            return x_last.copy()
        mu, sd = self.gp.predict(C)
        best = float(np.max(self.gp.predict(X)[0]))
        return C[int(np.argmax(expected_improvement(mu, sd, best)))]

    def recommend(self, h):
        if self.gp is None or len(self.X) != len(h["X"]):
            self._fit(h)
        return int(np.argmax(self.gp.predict(self.X)[0]))


class RandomWalkPolicy:
    """Trial and error with no rules at all: a random allowed step from the previous run."""

    def __init__(self, step, lo, hi, seed=0, fixed=None):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 23)

    def next(self, h):
        x_last = h["X"][-1]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        for _ in range(50):
            x = blo + self.rng.random(DIM) * (bhi - blo)
            if feasible(unscale(x))[0]:
                return x
        return x_last.copy()

    def recommend(self, h):
        return int(np.argmax(h["y_obs"]))


POLICIES = {"chemist": ChemistPolicy, "plain_bo": PlainBOPolicy, "random_walk": RandomWalkPolicy}


def run_baseline(system, method, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, max_step=None,
                 deactivation=0.0, check_every=0):
    """One campaign with a baseline policy, starting from the same 5 screening runs."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec, deactivation)
    x_base = scale(base)[0]
    step, lo, hi = step_scaled(max_step), np.zeros(DIM), np.ones(DIM)
    pol = POLICIES[method](step, lo, hi, seed=seed)
    while len(h["X"]) < budget:
        n = len(h["X"])
        if is_check_run(n, check_every):
            x, chk = x_base.copy(), True
        else:
            x, chk = pol.next(h), False
        _evaluate(vr, h, x, check=chk)
        _score_rec(vr, h, pol.recommend(h))
    return _pack(h, strategy=method, system=system, seed=seed, base=base)


# =================================================================================== true optimum (for scoring only)
_OPT_FILE = os.path.join(os.path.dirname(__file__), "optima.json")


def true_optimum(system, wcat=WCAT, a_spec=A_SPEC, recompute=False):
    """Global optimum of the TRUE profit at this catalyst load and full activity (differential
    evolution on the fast simulator, then polished with the exact LSODA solver), subject to the
    flow constraint.  Only used to score methods; the advisor never sees it.  Cached in
    smart/optima.json."""
    key = f"{system}|W={float(wcat):g}|a={float(a_spec):g}"
    cache = {}
    if os.path.exists(_OPT_FILE):
        with open(_OPT_FILE) as fh:
            cache = json.load(fh)
    if key in cache and not recompute:
        return cache[key]
    S = SYSTEMS[system]

    def fast(Xc):
        ph = unscale(Xc)
        P, VA, VB, VI, T = ph.T
        F, F0 = S.simulate(P, VA, VB, VI, wcat, T, truth=True, a_spec=a_spec)
        pen = 1e3 * np.maximum(0.0, MIN_TOTAL_FLOW - (VA + VB + VI))
        return S.profit(F, F0, P, wcat) - pen

    vr = VirtualReactor(S, 0, wcat, a_spec)
    best = None
    for s in range(2):
        res = differential_evolution(lambda Z: -fast(Z.T), [(0, 1)] * DIM, vectorized=True,
                                     seed=s, popsize=30, maxiter=300, tol=1e-10, polish=False)
        if best is None or res.fun < best.fun:
            best = res

    def exact(z):
        z = np.clip(z, 0, 1)
        if not feasible(unscale(z))[0]:
            return 1e6
        return -vr.true_profit(z)

    pol = minimize(exact, best.x, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=600))
    x = np.clip(pol.x, 0, 1)
    f = vr.true_profit(x)
    Xr = np.random.default_rng(123).random((4000, DIM))
    Xr = Xr[feasible(unscale(Xr))]
    ys = fast(Xr)
    cache[key] = dict(f_opt=f, x_opt=x.tolist(), x_opt_phys=unscale(x)[0].tolist(),
                      wcat=float(wcat), a_spec=float(a_spec),
                      f_median=float(np.median(ys)), frac_profitable=float(np.mean(ys > 0)))
    tmp = _OPT_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp, _OPT_FILE)
    return cache[key]
