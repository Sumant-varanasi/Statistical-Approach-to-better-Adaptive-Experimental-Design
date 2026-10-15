"""
The smart (physics-informed) reactor advisor.

Plain Bayesian optimisation (bo/) treats the reactor as a black box: it only sees one number
per experiment (the profit) and learns the landscape from scratch.  The advisor instead
knows the PHYSICS of the reactor and learns only what it does not know:

  KNOWN (built in)                         LEARNED from the experiments
  ---------------------------------------  ---------------------------------------------
  which reactions happen, stoichiometry    rate constants k_ref of every reaction
  mole balances along the packed bed       activation energies Ea (temperature response)
  Arrhenius temperature dependence         a correction for anything the physics misses
  ideal-gas partial pressures              (Gaussian process on the profit residual)
  equilibrium limit from thermodynamics
  the economics (prices)

Every experiment gives the full GC composition (a mole fraction per species), so one
experiment teaches the model about EVERY reaction at once, not just "profit was 12.3".

How the next experiment is chosen (each round):
  1. fit the kinetics to all GC readings so far (kinetics.py), with its uncertainty
  2. draw a handful of plausible versions of the chemistry consistent with the data
  3. for thousands of candidate operating points, simulate every version -> predicted
     profit (mean) and how much the versions disagree (physics uncertainty)
  4. add a Gaussian-process correction fitted to what the physics got wrong so far
  5. pick the candidate with the highest Expected Improvement (same EI as bo/)
So the model proposes experiments where the CHEMISTRY says profit should be high, or
where the plausible chemistries disagree most about something that matters.
"""
import json
import os
import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution, minimize
from .systems import (SYSTEMS, MODELS, LOW, HIGH, V_I, KEYS, unscale, scale, mfc_to_molar)
from .kinetics import KineticFit, GC_NOISE_REL, GC_NOISE_ABS
from bo.gp import GP
from bo.loop import LogInputs, initial_design
from bo.acquisition import expected_improvement

DIM = 5


# =================================================================================== the virtual reactor
class VirtualReactor:
    """Stands in for the real reactor + GC.  Uses the TRUE kinetics (with product
    inhibition, which the advisor's model does not have) and adds GC noise:
        y_GC = y_true * (1 + 1 % noise) + 0.05 % absolute noise, renormalised
    (typical repeatability of an online GC with an internal standard).
    The operator never sees profit directly: it is computed from the GC reading using the
    inert as an internal standard, exactly as the advisor tool does with real data."""

    def __init__(self, system, seed=0, gc_rel=GC_NOISE_REL, gc_abs=GC_NOISE_ABS):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.rng = np.random.default_rng(50_000 + seed)
        self.gc_rel, self.gc_abs = gc_rel, gc_abs

    def true_flows(self, x):
        P, VA, VB, T, W = unscale(x)[0]
        F, F0 = self.S.simulate(P, VA, VB, V_I, W, T, truth=True, method="lsoda")
        return F, F0, (P, W)

    def true_profit(self, x):
        F, F0, (P, W) = self.true_flows(x)
        return float(self.S.profit(F, F0, [P], [W])[0])

    def run(self, x):
        F, F0, (P, W) = self.true_flows(x)
        y = F / F.sum(1, keepdims=True)
        y = y * (1 + self.gc_rel * self.rng.standard_normal(y.shape)) \
            + self.gc_abs * self.rng.standard_normal(y.shape)
        y = np.maximum(y, 0.0)
        y /= y.sum(1, keepdims=True)
        return dict(gc=y[0], profit=float(profit_from_gc(self.S, unscale(x)[0], y[0])[0]),
                    profit_true=float(self.S.profit(F, F0, [P], [W])[0]))


def profit_from_gc(S, phys, y):
    """Profit from settings + a GC reading (inert = internal standard).  This is all the
    operator has to provide: what they set, and what the GC printed."""
    phys, y = np.atleast_2d(phys), np.atleast_2d(y)
    P, VA, VB, T, W = phys.T
    F = S.flows_from_gc(y, VA, VB, V_I)
    F0 = np.zeros_like(F)
    F0[:, 0], F0[:, 1], F0[:, -1] = mfc_to_molar(VA), mfc_to_molar(VB), mfc_to_molar(V_I)
    return S.profit(F, F0, P, W)


def profit_noise_sd(S, phys, y, n=64, seed=0):
    """How much GC noise alone moves the profit computed from a reading (Monte Carlo with
    the GC error model).  Profit is a small difference of large numbers, so 1 % GC noise
    becomes several money units per hour; the correction GP must not mistake that for a
    real effect."""
    rng = np.random.default_rng(seed)
    phys, y = np.atleast_2d(phys), np.atleast_2d(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        Yn = y[i] * (1 + GC_NOISE_REL * rng.standard_normal((n, y.shape[1])))             + GC_NOISE_ABS * rng.standard_normal((n, y.shape[1]))
        Yn = np.maximum(Yn, 0)
        Yn /= Yn.sum(1, keepdims=True)
        out[i] = profit_from_gc(S, np.repeat(phys[i:i + 1], n, 0), Yn).std()
    return out


class SmoothGP(GP):
    """GP for the physics correction: length-scales of at least 0.15 (in scaled units).
    What the physics gets wrong (here: product inhibition) changes slowly across the box;
    anything wigglier than that is measurement noise, not chemistry."""
    LS_BOUNDS = (np.log(0.15), np.log(20.0))


# =================================================================================== the advisor
class SmartAdvisor:
    """
    adv = SmartAdvisor('complex')
    adv.update(X, gc, profit)          X scaled (n x 5), gc (n x ns), measured profit (n,)
    x, info = adv.suggest(rng)         next experiment (scaled) and why
    adv.predict(Xc)                    predicted profit mean and sd at candidates
    adv.recommend()                    index of the best tested experiment (by prediction)

    lo, hi      optional safety box in scaled units (e.g. never above 700 K)
    fixed       {knob index: scaled value} knobs the operator cannot change
    residual_gp correct the physics with a GP on the profit residual (on by default)
    """

    def __init__(self, system, seed=0, n_theta=6, lo=None, hi=None, fixed=None,
                 residual_gp=True):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.kin = KineticFit(self.S, seed=seed)
        self.n_theta = n_theta
        self.rng = np.random.default_rng(seed)
        self.lo = np.zeros(DIM) if lo is None else np.asarray(lo, float)
        self.hi = np.ones(DIM) if hi is None else np.asarray(hi, float)
        self.fixed = dict(fixed or {})
        self.use_gp = residual_gp
        self.gp = None
        self.seed = seed

    # ------------------------------------------------------------------ learning
    def update(self, X, gc, profit):
        self.X = np.atleast_2d(np.asarray(X, float))
        self.gc = np.atleast_2d(gc)
        self.y = np.asarray(profit, float)
        n = len(self.y)
        n_starts = None if self.kin.n_fits == 0 else (3 if n % 5 == 0 else 1)
        self.kin.fit(unscale(self.X), self.gc, n_starts=n_starts)
        self.thetas = self.kin.samples(self.n_theta, self.rng)
        self.phys_at_data = self._physics(self.X, self.kin.theta[None])[0]
        self.resid = self.y - self.phys_at_data
        if self.use_gp and n >= 4:
            # the GP's noise is not allowed to drop below what the GC noise implies
            s = max(self.resid.std(), 1e-6)
            nf = np.mean(profit_noise_sd(self.S, unscale(self.X), self.gc) ** 2) / s ** 2
            gp = SmoothGP(seed=self.seed, noise_bounds=(min(nf, 0.5), max(1.0, 2 * nf)))
            self.gp = LogInputs(gp, (0, 4), LOW, HIGH).fit(self.X, self.resid)
        return self

    def _physics(self, Xc, thetas):
        """Profit predicted by the physics model for every (theta, candidate) pair: (m, n)."""
        m, n = len(thetas), len(Xc)
        ph = np.tile(unscale(Xc), (m, 1))
        P, VA, VB, T, W = ph.T
        F, F0 = self.S.simulate(P, VA, VB, V_I, W, T, theta=np.repeat(thetas, n, 0))
        pr = self.S.profit(F, F0, P, W).reshape(m, n)
        return np.nan_to_num(pr, nan=-1e3, posinf=-1e3, neginf=-1e3)

    def predict(self, Xc, parts=False):
        Xc = np.atleast_2d(Xc)
        pr = self._physics(Xc, self.thetas)
        mu_p = pr[0]                                   # best-estimate chemistry
        sd_p = pr.std(0)                               # disagreement between plausible chemistries
        if self.gp is not None:
            mu_r, sd_r = self.gp.predict(Xc)
        else:
            mu_r, sd_r = np.zeros(len(Xc)), np.zeros(len(Xc))
        mu, sd = mu_p + mu_r, np.sqrt(sd_p ** 2 + sd_r ** 2 + 1e-9)
        if parts:
            return mu, sd, dict(mu_phys=mu_p, sd_phys=sd_p, mu_corr=mu_r, sd_corr=sd_r)
        return mu, sd

    def recommend(self):
        """Index of the tested experiment with the best PREDICTED profit (a single lucky
        GC reading cannot fool it)."""
        return int(np.argmax(self.predict(self.X)[0]))

    # ------------------------------------------------------------------ choosing the next experiment
    def _clip(self, C):
        C = np.clip(C, self.lo, self.hi)
        for i, v in self.fixed.items():
            C[:, i] = v
        return C

    def suggest(self, rng=None):
        rng = rng or self.rng
        mu_data = self.predict(self.X)[0]
        best = mu_data.max()
        # stage 1: space-filling candidates + points around the best experiments so far
        sob = self.lo + qmc.Sobol(DIM, scramble=True, seed=rng.integers(1 << 31)).random(768) * (self.hi - self.lo)
        top = self.X[np.argsort(mu_data)[-5:]]
        loc = top[rng.integers(len(top), size=256)] + rng.normal(0, 0.05, (256, DIM))
        C1 = self._clip(np.vstack([sob, loc]))
        mu1, sd1 = self.predict(C1)
        ei1 = expected_improvement(mu1, sd1, best)
        # stage 2: zoom in around the 8 most promising candidates
        seeds = C1[np.argsort(ei1)[-8:]]
        C2 = self._clip(seeds[rng.integers(8, size=384)] + rng.normal(0, 0.015, (384, DIM)))
        mu2, sd2 = self.predict(C2)
        ei2 = expected_improvement(mu2, sd2, best)
        C, mu, sd, ei = (np.vstack([C1, C2]), np.r_[mu1, mu2], np.r_[sd1, sd2], np.r_[ei1, ei2])
        j = int(np.argmax(ei))
        k = int(np.argmax(mu))
        return C[j], dict(pred=mu[j], sd=sd[j], ei=ei[j], best_pred=best,
                          model_opt=C[k], model_opt_pred=mu[k], model_opt_sd=sd[k])


# =================================================================================== optimisation runs
def run_smart(system, budget=30, seed=0, n_init=8, verbose=False, model=None, callback=None):
    """One optimisation campaign with the smart advisor on the virtual reactor.  Starts
    from the same Latin-hypercube design as the bo/ strategies with the same seed.
    model: name in systems.MODELS to give the advisor a deliberately WRONG reaction network
    (stress test); default = the right network (still without the product inhibition).
    callback(run_number, x, result, info) is called after every advisor-chosen experiment."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed)
    adv = SmartAdvisor(MODELS[model] if model else S, seed=seed)
    rng = np.random.default_rng(seed)
    X, G, Y, Yt, rec = [], [], [], [], []
    for x in initial_design(n_init, DIM, seed):
        r = vr.run(x)
        X.append(x); G.append(r["gc"]); Y.append(r["profit"]); Yt.append(r["profit_true"])
        rec.append(Yt[int(np.argmax(Y))])
    info_log = []
    while len(X) < budget:
        adv.update(np.array(X), np.array(G), np.array(Y))
        rec[-1] = Yt[adv.recommend()]
        x, info = adv.suggest(rng)
        r = vr.run(x)
        X.append(x); G.append(r["gc"]); Y.append(r["profit"]); Yt.append(r["profit_true"])
        info_log.append(info)
        if len(X) == budget:
            adv.update(np.array(X), np.array(G), np.array(Y))
        rec.append(Yt[adv.recommend()])
        if callback:
            callback(len(X), x, r, info)
        if verbose:
            print(f"smart {system} run {len(X):2d}: measured {r['profit']:7.2f}  "
                  f"predicted {info['pred']:7.2f} +- {info['sd']:.2f}   recommended (true) {rec[-1]:7.2f}")
    return dict(strategy="smart" if not model else f"smart_{model}", system=system, seed=seed,
                X=np.array(X), y_obs=np.array(Y), gc=np.array(G),
                y_true=np.array(Yt), rec=np.array(rec), theta=adv.kin.theta, info=info_log,
                advisor=adv)


class ProfitProblem:
    """Adapter so the existing bo/ strategies (bo.loop.run) can optimise the same virtual
    reactor: they see only the measured profit, no GC composition, no physics."""

    def __init__(self, system, seed=0):
        self.S = SYSTEMS[system]
        self.name = f"profit_{system}"
        self.vr = VirtualReactor(self.S, seed)
        self.dim = DIM
        self.noise_sd = 0.5          # nominal: tells bo.loop to recommend by model mean
        self.constraint = None
        o = true_optimum(system)
        self.f_opt, self.f_ref = o["f_opt"], o["f_median"]

    def true(self, x):
        return self.vr.true_profit(x)

    def __call__(self, x):
        r = self.vr.run(x)
        return r["profit"], r["profit_true"]


def run_ofat(system, budget=30, seed=0, start=(10.0, 20.0, 20.0, 600.0, 1.0)):
    """'Trial and error' the way it is usually done by hand: one factor at a time.
    Start from a sensible-looking setting (the project's base case), then for each knob in
    turn try 5 levels across its range with the others held fixed, keep the best, move on;
    after one pass, repeat with finer steps around the best.  Uses measured profit only."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed)
    rng = np.random.default_rng(seed)
    order = [3, 4, 0, 2, 1]                 # T, Wcat, P, V_B, V_A: what an operator would try first
    cur = scale(np.array(start))[0]
    X, Y, Yt, rec = [], [], [], []

    def ev(x):
        r = vr.run(x)
        X.append(x.copy()); Y.append(r["profit"]); Yt.append(r["profit_true"])
        rec.append(Yt[int(np.argmax(Y))])
        return r["profit"]

    best_y = ev(cur)
    width = 1.0
    while len(X) < budget:
        for i in order:
            lv = np.clip(cur[i] + width * np.linspace(-0.5, 0.5, 5) if width < 1 else np.linspace(0, 1, 5), 0, 1)
            lv = lv + (rng.uniform(-0.02, 0.02, 5) if width < 1 else 0)
            for v in np.clip(lv, 0, 1):
                if len(X) >= budget:
                    break
                x = cur.copy(); x[i] = v
                y = ev(x)
                if y > best_y:
                    best_y, best_x = y, x
            cur = X[int(np.argmax(Y))].copy()
        width = 0.4 if width == 1.0 else width / 2
    return dict(strategy="ofat", system=system, seed=seed, X=np.array(X), y_obs=np.array(Y),
                y_true=np.array(Yt), rec=np.array(rec))


# =================================================================================== true optimum (for scoring only)
_OPT_FILE = os.path.join(os.path.dirname(__file__), "optima.json")


def true_optimum(system, recompute=False):
    """Global optimum of the TRUE profit (differential evolution on the fast simulator, then
    polished with the exact LSODA solver).  Only used to score methods; the advisor never
    sees it.  Cached in smart/optima.json."""
    cache = {}
    if os.path.exists(_OPT_FILE):
        with open(_OPT_FILE) as fh:
            cache = json.load(fh)
    if system in cache and not recompute:
        return cache[system]
    S = SYSTEMS[system]

    def fast(Xc):
        ph = unscale(Xc)
        P, VA, VB, T, W = ph.T
        F, F0 = S.simulate(P, VA, VB, V_I, W, T, truth=True)
        return S.profit(F, F0, P, W)

    vr = VirtualReactor(S)
    best = None
    for s in range(2):
        res = differential_evolution(lambda Z: -fast(Z.T), [(0, 1)] * DIM, vectorized=True,
                                     seed=s, popsize=30, maxiter=300, tol=1e-10, polish=False)
        if best is None or res.fun < best.fun:
            best = res
    pol = minimize(lambda z: -vr.true_profit(np.clip(z, 0, 1)), best.x, method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-6, maxiter=600))
    x = np.clip(pol.x, 0, 1)
    f = vr.true_profit(x)
    ys = fast(np.random.default_rng(123).random((4000, DIM)))
    cache[system] = dict(f_opt=f, x_opt=x.tolist(), x_opt_phys=unscale(x)[0].tolist(),
                         f_median=float(np.median(ys)),
                         frac_profitable=float(np.mean(ys > 0)))
    tmp = _OPT_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp, _OPT_FILE)
    return cache[system]
