"""
The smart (physics-informed) reactor advisor, and the baselines it is compared with.

Plain Bayesian optimisation (bo/) treats the reactor as a black box: it only sees one number
per experiment (the profit) and learns the landscape from scratch.  The advisor instead
knows the PHYSICS of the reactor and learns only what it does not know:

  KNOWN (built in)                         LEARNED from the experiments
  ---------------------------------------  ---------------------------------------------
  which reactions happen, stoichiometry    rate constants k_ref of every reaction
  mole balances over the catalyst surface  activation energies Ea (temperature response)
  Arrhenius temperature dependence         a correction for anything the physics misses
  ideal-gas partial pressures              (Gaussian process on the profit residual)
  equilibrium limit from thermodynamics
  the economics (prices)

Every experiment gives the full GC composition (a mole fraction per species), so one
experiment teaches the model about EVERY reaction at once, not just "profit was 12.3".

How a campaign goes (same for every method, because that is how the reactor is run):
  * the catalyst is loaded once (WCAT) and never changed
  * the operator's 5 screening runs come first (systems.screening_runs)
  * from then on every suggested run must lie within MAX_STEP of the previous run for every
    knob, inside the safety box, and leave enough total flow for the GC

How the advisor chooses the next run:
  1. fit the kinetics to all GC readings so far (kinetics.py), with its uncertainty
  2. draw a handful of plausible versions of the chemistry consistent with the data
  3. PLAN: find where the model expects the best profit over the whole box (x*)
  4. among the runs reachable in one step, score candidates by Expected Improvement, with
     the step toward x* always among them; the chosen run is usually the biggest viable
     step uphill, and the advisor can say how many runs away x* is
  5. a Gaussian-process correction, fitted to what the physics got wrong so far, is added
     to every prediction (with a noise floor from the GC error)
"""
import json
import os
import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution, minimize
from .systems import (SYSTEMS, MODELS, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, WCAT, A_SPEC, DIM,
                      unscale, scale, physical, feasible, step_scaled, base_case, screening_runs)
from .kinetics import KineticFit, GC_NOISE_REL, GC_NOISE_ABS
from bo.gp import GP
from bo.loop import LogInputs, copula_transform
from bo.acquisition import expected_improvement


# =================================================================================== the virtual reactor
class VirtualReactor:
    """Stands in for the real reactor + GC.  Uses the TRUE kinetics (with product
    inhibition, which the advisor's model does not have) and adds GC noise:
        y_GC = y_true * (1 + 1 % noise) + 0.05 % absolute noise, renormalised
    (typical repeatability of an online GC with an internal standard).
    The operator never sees profit directly: it is computed from the GC reading using the
    inert as an internal standard, exactly as the advisor tool does with real data."""

    def __init__(self, system, seed=0, wcat=WCAT, a_spec=A_SPEC, gc_rel=GC_NOISE_REL, gc_abs=GC_NOISE_ABS):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.rng = np.random.default_rng(50_000 + seed)
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.gc_rel, self.gc_abs = gc_rel, gc_abs

    def true_flows(self, x):
        P, VA, VB, VI, T = unscale(x)[0]
        F, F0 = self.S.simulate(P, VA, VB, VI, self.wcat, T, truth=True, method="lsoda", a_spec=self.a_spec)
        return F, F0, P

    def true_profit(self, x):
        F, F0, P = self.true_flows(x)
        return float(self.S.profit(F, F0, [P], [self.wcat])[0])

    def run(self, x):
        F, F0, P = self.true_flows(x)
        y = F / F.sum(1, keepdims=True)
        y = y * (1 + self.gc_rel * self.rng.standard_normal(y.shape)) \
            + self.gc_abs * self.rng.standard_normal(y.shape)
        y = np.maximum(y, 0.0)
        y /= y.sum(1, keepdims=True)
        ph6 = physical(x, self.wcat)
        XA, SC, _ = kpis_from_gc(self.S, ph6, y)
        return dict(gc=y[0], profit=float(profit_from_gc(self.S, ph6, y)[0]),
                    profit_true=float(self.S.profit(F, F0, [P], [self.wcat])[0]),
                    X_A=float(XA[0]), S_C=float(SC[0]))


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


def sobol_in(lo, hi, n, rng):
    return lo + qmc.Sobol(DIM, scramble=True, seed=int(rng.integers(1 << 31))).random(n) * (hi - lo)


def keep_feasible(C):
    return C[feasible(unscale(C))]


# =================================================================================== the advisor
class SmartAdvisor:
    """
    adv = SmartAdvisor('complex', wcat=1.0, a_spec=100)
    adv.update(X, gc, profit)          X scaled (n x 5), gc (n x ns), measured profit (n,)
    x, info = adv.suggest(x_prev)      next run (scaled), reachable from x_prev, and why
    adv.predict(Xc)                    predicted profit mean and sd at candidates
    adv.recommend()                    index of the best tested run (by prediction)
    adv.plan()                         where the model expects the best profit (x*, mean, sd)

    wcat, a_spec  catalyst mass (g) and BET area (m2/g) of the loaded bed
    max_step      largest change per knob between consecutive runs (physical units)
    lo, hi        optional safety box in scaled units (e.g. never above 700 K)
    fixed         {knob index: scaled value} knobs the operator cannot change
    residual_gp   correct the physics with a GP on the profit residual (on by default)
    """

    def __init__(self, system, wcat=WCAT, a_spec=A_SPEC, seed=0, n_theta=6, lo=None, hi=None,
                 fixed=None, max_step=None, residual_gp=True):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.kin = KineticFit(self.S, a_spec=self.a_spec, seed=seed)
        self.n_theta = n_theta
        self.rng = np.random.default_rng(seed)
        self.lo = np.zeros(DIM) if lo is None else np.asarray(lo, float)
        self.hi = np.ones(DIM) if hi is None else np.asarray(hi, float)
        self.fixed = dict(fixed or {})
        self.step = step_scaled(max_step)
        self.use_gp = residual_gp
        self.gp = None
        self.seed = seed

    # ------------------------------------------------------------------ learning
    def update(self, X, gc, profit, wcat_rows=None):
        """wcat_rows: catalyst mass of each logged run (defaults to the campaign's wcat);
        the physics uses each run's own loading, so a log spanning two loadings is fine."""
        self.X = np.atleast_2d(np.asarray(X, float))
        self.gc = np.atleast_2d(gc)
        self.y = np.asarray(profit, float)
        n = len(self.y)
        self.W = np.broadcast_to(np.asarray(self.wcat if wcat_rows is None else wcat_rows, float), (n,)).copy()
        ph6 = physical(self.X, self.W)
        n_starts = None if self.kin.n_fits == 0 else (3 if n % 5 == 0 else 1)
        self.kin.fit(ph6, self.gc, n_starts=n_starts)
        self.thetas = self.kin.samples(self.n_theta, self.rng)
        self.phys_at_data = self._physics(self.X, self.kin.theta[None], W=self.W)[0]
        self.resid = self.y - self.phys_at_data
        if self.use_gp and n >= 4:
            # the GP's noise is not allowed to drop below what the GC noise implies
            s = max(self.resid.std(), 1e-6)
            nf = np.mean(profit_noise_sd(self.S, ph6, self.gc) ** 2) / s ** 2
            gp = SmoothGP(seed=self.seed, noise_bounds=(min(nf, 0.5), max(1.0, 2 * nf)))
            self.gp = LogInputs(gp, (0,), LOW, HIGH).fit(self.X, self.resid)     # GP sees log P
        return self

    def _physics(self, Xc, thetas, W=None):
        """Profit predicted by the physics model for every (theta, candidate) pair: (m, n)."""
        m, n = len(thetas), len(Xc)
        ph = np.tile(unscale(Xc), (m, 1))
        P, VA, VB, VI, T = ph.T
        Wr = np.tile(np.broadcast_to(np.asarray(self.wcat if W is None else W, float), (n,)), m)
        F, F0 = self.S.simulate(P, VA, VB, VI, Wr, T, theta=np.repeat(thetas, n, 0), a_spec=self.a_spec)
        with np.errstate(invalid="ignore"):
            pr = self.S.profit(F, F0, P, Wr).reshape(m, n)
        pr[np.isinf(pr)] = np.nan                      # NaN = this chemistry gave no answer here
        return pr

    def predict(self, Xc, parts=False):
        Xc = np.atleast_2d(Xc)
        pr = self._physics(Xc, self.thetas)
        mu_p = np.nan_to_num(pr[0], nan=-1e3)          # best-estimate chemistry
        with np.errstate(invalid="ignore"):
            sd_p = np.nan_to_num(np.nanstd(pr, 0), nan=0.0)   # disagreement between plausible chemistries
        if self.gp is not None:
            mu_r, sd_r = self.gp.predict(Xc)
        else:
            mu_r, sd_r = np.zeros(len(Xc)), np.zeros(len(Xc))
        mu, sd = mu_p + mu_r, np.sqrt(sd_p ** 2 + sd_r ** 2 + 1e-9)
        if parts:
            return mu, sd, dict(mu_phys=mu_p, sd_phys=sd_p, mu_corr=mu_r, sd_corr=sd_r)
        return mu, sd

    def recommend(self):
        """Index of the tested run with the best PREDICTED profit (a single lucky GC reading
        cannot fool it)."""
        return int(np.argmax(self.predict(self.X)[0]))

    # ------------------------------------------------------------------ planning
    def _pin(self, C):
        C = np.clip(C, self.lo, self.hi)
        for i, v in self.fixed.items():
            C[:, i] = v
        return C

    def plan(self, rng=None):
        """Where the model expects the best profit over the WHOLE safety box (not just one
        step away): x*, its predicted profit and uncertainty."""
        rng = rng or self.rng
        # space-filling candidates, plus the runs already done and their neighbourhood (the
        # profitable window can be narrow enough for 1024 Sobol points to miss it)
        mu_data = self.predict(self.X)[0]
        x_best = self.X[int(np.argmax(mu_data))]
        C1 = np.vstack([sobol_in(self.lo, self.hi, 1024, rng), self.X,
                        x_best + rng.normal(0, 0.5, (64, DIM)) * self.step])
        C1 = keep_feasible(self._pin(C1))
        mu1, _ = self.predict(C1)
        top = C1[np.argsort(mu1)[-5:]]
        C2 = keep_feasible(self._pin(top[rng.integers(len(top), size=256)] + rng.normal(0, 0.03, (256, DIM))))
        mu2, sd2 = self.predict(C2)
        C, mu, sd = np.vstack([C1, C2]), np.r_[mu1, np.array(mu2)], np.r_[np.full(len(C1), np.nan), sd2]
        k = int(np.argmax(mu))
        if np.isnan(sd[k]):
            sd[k] = self.predict(C[k:k + 1])[1][0]
        return C[k], float(mu[k]), float(sd[k])

    def steps_away(self, x_from, x_to):
        """How many runs it takes to get from x_from to x_to under the step limits."""
        return int(np.ceil(np.max(np.abs(np.asarray(x_to) - np.asarray(x_from)) / self.step - 1e-9)))

    # ------------------------------------------------------------------ choosing the next run
    def suggest(self, x_prev=None, rng=None):
        """The next run, reachable from x_prev (default: the last logged run) in one step."""
        rng = rng or self.rng
        x_prev = np.asarray(self.X[-1] if x_prev is None else x_prev, float)
        x_star, mu_star, sd_star = self.plan(rng)
        blo, bhi = step_box(x_prev, self.step, self.lo, self.hi, self.fixed)
        waypoint = np.clip(x_star, blo, bhi)               # the biggest allowed step toward x*
        mu_data = self.predict(self.X)[0]
        best = float(mu_data.max())
        x_best = self.X[int(np.argmax(mu_data))]
        C = np.vstack([
            sobol_in(blo, bhi, 512, rng),
            waypoint[None],
            np.clip(waypoint + rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi),
            np.clip(x_best + rng.normal(0, 0.3, (64, DIM)) * self.step, blo, bhi),
        ])
        C = self._pin(C)
        ok = feasible(unscale(C))
        if not ok.any():                                   # nothing reachable is feasible: stay put
            return x_prev.copy(), dict(pred=best, sd=0.0, ei=0.0, best_pred=best, model_opt=x_star,
                                       model_opt_pred=mu_star, model_opt_sd=sd_star,
                                       steps_away=self.steps_away(x_prev, x_star), toward_opt=False)
        C = C[ok]
        mu, sd = self.predict(C)
        ei = expected_improvement(mu, sd, best)
        j = int(np.argmax(ei))
        mode = "improve"
        # Purposeful walk: when nothing within reach looks like an improvement but the model is
        # confident a much better region exists further away (e.g. the best run so far sits in
        # a local hollow), take the biggest allowed step toward it instead of dithering.
        far_gain = mu_star - 1.64 * sd_star - best
        if far_gain > 0 and ei[j] < 0.1 * far_gain:
            # make a full step on the knob that is furthest behind (the bottleneck, e.g. the
            # temperature), and set the other knobs to whatever is best at that step
            dist = np.abs(x_star - x_prev) / self.step
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
    return dict(X=[], gc=[], y_obs=[], y_true=[], X_A=[], S_C=[], rec=[])


def _evaluate(vr, h, x):
    r = vr.run(x)
    h["X"].append(np.asarray(x, float).copy()); h["gc"].append(r["gc"])
    h["y_obs"].append(r["profit"]); h["y_true"].append(r["profit_true"])
    h["X_A"].append(r["X_A"]); h["S_C"].append(r["S_C"])
    h["rec"].append(h["y_true"][int(np.argmax(h["y_obs"]))])
    return r


def start_campaign(system, seed=0, wcat=WCAT, a_spec=A_SPEC):
    """The virtual reactor and the operator's 5 screening runs (shared by every method)."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed, wcat, a_spec)
    h = _new_history()
    base = base_case(seed)
    for row in screening_runs(base):
        _evaluate(vr, h, scale(row)[0])
    return S, vr, h, base


def _pack(h, **extra):
    out = {k: np.array(v) for k, v in h.items()}
    out.update(extra)
    return out


def run_smart(system, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, verbose=False, model=None, callback=None):
    """One campaign with the smart advisor on the virtual reactor.
    model: name in systems.MODELS to give the advisor a deliberately WRONG reaction network
    (stress test); default = the right network (still without the product inhibition).
    callback(run_number, x, result, info) is called after every advisor-chosen run."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec)
    adv = SmartAdvisor(MODELS[model] if model else S, wcat=wcat, a_spec=a_spec, seed=seed)
    rng = np.random.default_rng(seed)
    info_log = []
    while len(h["X"]) < budget:
        adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]))
        h["rec"][-1] = h["y_true"][adv.recommend()]
        x, info = adv.suggest(h["X"][-1], rng)
        r = _evaluate(vr, h, x)
        info_log.append(info)
        if len(h["X"]) == budget:
            adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]))
        h["rec"][-1] = h["y_true"][adv.recommend()]
        if callback:
            callback(len(h["X"]), x, r, info)
        if verbose:
            print(f"smart {system} run {len(h['X']):2d}: measured {r['profit']:7.2f}  predicted "
                  f"{info['pred']:7.2f} +- {info['sd']:.2f}   recommended (true) {h['rec'][-1]:7.2f}")
    return _pack(h, strategy="smart" if not model else f"smart_{model}", system=system, seed=seed,
                 base=base, theta=adv.kin.theta, info=info_log, advisor=adv)


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


def run_baseline(system, method, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, max_step=None):
    """One campaign with a baseline policy, starting from the same 5 screening runs."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec)
    step, lo, hi = step_scaled(max_step), np.zeros(DIM), np.ones(DIM)
    pol = POLICIES[method](step, lo, hi, seed=seed)
    while len(h["X"]) < budget:
        x = pol.next(h)
        _evaluate(vr, h, x)
        h["rec"][-1] = h["y_true"][pol.recommend(h)]
    return _pack(h, strategy=method, system=system, seed=seed, base=base)


# =================================================================================== true optimum (for scoring only)
_OPT_FILE = os.path.join(os.path.dirname(__file__), "optima.json")


def true_optimum(system, wcat=WCAT, a_spec=A_SPEC, recompute=False):
    """Global optimum of the TRUE profit at this catalyst load (differential evolution on the
    fast simulator, then polished with the exact LSODA solver), subject to the flow
    constraint.  Only used to score methods; the advisor never sees it.  Cached in
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
