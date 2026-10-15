"""
The optimisation loop and every strategy we compare.

    run(strategy_name, problem, budget, seed)  ->  history dict

Strategies (see STRATEGIES):
    random      uniform random search                                  (baseline)
    lhs         Latin-hypercube space-filling design of the whole budget (baseline, "grid-like")
    simplex     Nelder-Mead from the centre of the box                  (classic local search)
    bo_v1       the original bo_5knobs.py method: RBF, fixed length-scale 0.3, EI on 5,000
                random candidates, random start                         (our own starting point)
    gp_ei       fitted ARD Matern GP + Expected Improvement
    gp_ucb      fitted ARD Matern GP + Upper Confidence Bound
    gp_pi       fitted ARD Matern GP + Probability of Improvement
    gp_ts       fitted ARD Matern GP + Thompson Sampling
    nn_ei       deep-ensemble neural network + Expected Improvement
    nn_ts       deep-ensemble neural network + Thompson Sampling
    gp_ei_q4    gp_ei, but 4 experiments chosen at once per round (kriging believer):
                same number of experiments, a quarter of the rounds

All BO strategies with the same seed start from the SAME 10-point Latin-hypercube design,
so differences between them are due to the method, not the luck of the start (this also
lets us use paired statistical tests).

What is recorded for every evaluation t:
    y_obs[t]   value the optimiser saw (includes measurement noise if the problem is noisy)
    y_true[t]  noise-free value of that experiment
    rec[t]     noise-free value of the point the method would RECOMMEND after t experiments:
               the best observed point if there is no noise; with noise, BO recommends the
               point with the best GP-predicted mean (a noisy lucky reading can't fool it).
"""
import time
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from .gp import GP, GPEnsemble
from .nn import NNEnsemble
from .acquisition import propose, propose_constrained, expected_improvement

STRATEGIES = {
    "random":   dict(kind="random", label="Random search"),
    "lhs":      dict(kind="lhs", label="Latin hypercube (space-filling)"),
    "simplex":  dict(kind="simplex", label="Nelder-Mead simplex"),
    "bo_v1":    dict(kind="bo_v1", label="BO v1 (fixed RBF, EI)"),
    "gp_ei":    dict(kind="bo", model="gp", acq="ei", label="GP + EI"),
    "gp_ucb":   dict(kind="bo", model="gp", acq="ucb", label="GP + UCB"),
    "gp_pi":    dict(kind="bo", model="gp", acq="pi", label="GP + PI"),
    "gp_ts":    dict(kind="bo", model="gp", acq="ts", label="GP + Thompson"),
    "nn_ei":    dict(kind="bo", model="nn", acq="ei", label="NN ensemble + EI"),
    "nn_ts":    dict(kind="bo", model="nn", acq="ts", label="NN ensemble + Thompson"),
    "gp_ei_q4": dict(kind="bo", model="gp", acq="ei", q=4, label="GP + EI, batch of 4"),
    # constrained BO: a second GP models the product-spec constraint (only meaningful on a
    # Problem created with constraint=...; other strategies simply ignore the constraint
    # while searching, but are still only credited with operating points that meet it)
    "cgp_ei":   dict(kind="bo", model="gp", acq="cei", label="Constrained GP + EI"),
    # GP sees catalyst mass and pressure (and optionally the flows) on a LOG scale
    "gp_ei_logw": dict(kind="bo", model="gp", acq="ei", warp_in=(0, 4), label="GP + EI, log P & W_cat"),
    "gp_ei_logall": dict(kind="bo", model="gp", acq="ei", warp_in=(0, 1, 2, 4), label="GP + EI, log P, flows & W_cat"),
    # same strategies but WITHOUT the output transform, to measure what it adds
    "gp_ei_raw": dict(kind="bo", model="gp", acq="ei", warp=False, label="GP + EI, no output transform"),
    # v3 additions: a trust region around the incumbent (TuRBO-1 rules) against getting stuck,
    # and a likelihood-weighted ensemble of GP hyper-parameter optima for honest uncertainty
    "gp_ei_tr": dict(kind="bo", model="gp", acq="ei", tr=True, label="GP + EI, trust region"),
    "gpe_ei": dict(kind="bo", model="gpe", acq="ei", label="GP ensemble + EI"),
    "gpe_ei_tr": dict(kind="bo", model="gpe", acq="ei", tr=True, label="GP ensemble + EI, trust region"),
}


class TrustRegion:
    """TuRBO-1 style trust region in the scaled box: a hyper-rectangle of side L around the
    incumbent, sides scaled by the GP length-scales (normalised to geometric mean 1).  L doubles
    after `tol_s` consecutive improvements, halves after `tol_f` consecutive failures, and is
    reset to L0 when it falls below Lmin (the data are kept: a soft restart)."""

    def __init__(self, dim, L0=0.8, Lmin=0.05, Lmax=1.6, tol_s=3, tol_f=3):
        self.dim, self.L0, self.L, self.Lmin, self.Lmax = dim, L0, L0, Lmin, Lmax
        self.tol_s, self.tol_f = tol_s, tol_f
        self.succ = self.fail = self.restarts = 0

    def box(self, center, ls=None):
        w = np.ones(self.dim)
        if ls is not None:
            ls = np.asarray(ls, float)
            w = np.clip(ls / np.exp(np.mean(np.log(ls))), 0.25, 4.0)
        half = 0.5 * self.L * w
        return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)

    def update(self, improved):
        if improved:
            self.succ, self.fail = self.succ + 1, 0
        else:
            self.fail, self.succ = self.fail + 1, 0
        if self.succ >= self.tol_s:
            self.L, self.succ = min(2 * self.L, self.Lmax), 0
        if self.fail >= self.tol_f:
            self.L, self.fail = self.L / 2, 0
        if self.L < self.Lmin:
            self.L, self.restarts = self.L0, self.restarts + 1


def _length_scales(model):
    """Fitted length-scales of a GP, GP ensemble or a LogInputs wrapper around one (or None)."""
    for obj in (model, getattr(model, "m", None)):
        ls = getattr(obj, "ls", None)
        if ls is not None:
            return np.asarray(ls, float)
    return None


def copula_transform(y):
    """Rank-based ('Gaussian copula') output transform:  y -> Phi^-1(rank / (n + 1)).
    It keeps the ORDER of the experiments but squashes extreme values, so a few very bad
    experiments (huge losses) no longer dominate the GP fit.  The GP then spends its
    accuracy where it matters for optimisation: near the best points.  Because the
    transform is monotone, 'best predicted' is unchanged."""
    from scipy.stats import norm, rankdata
    r = rankdata(y)                      # ties share the average rank
    return norm.ppf(r / (len(y) + 1))

N_INIT = 10


class LogInputs:
    """Wraps a surrogate so it sees some inputs on a log scale.

    The optimiser still searches the same box (scaled 0-1); only the model's view changes.
    Knob i with physical range [lo, hi] is mapped  x -> (log(lo + x (hi - lo)) - log lo) / (log hi - log lo).
    Catalyst mass (0.05-5 g) and pressure (1-100 bar) both span two orders of magnitude, and
    reaction extent depends on them roughly logarithmically, so equal steps in log units are
    far more alike than equal steps in grams or bar."""

    def __init__(self, model, idx, low, high):
        self.m, self.idx = model, list(idx)
        self.lo, self.hi = np.log(low[self.idx]), np.log(high[self.idx])
        self.low, self.high = low[self.idx], high[self.idx]
        self.smooth = getattr(model, "smooth", True)

    def g(self, X):
        X = np.array(np.atleast_2d(X), float)
        phys = self.low + X[:, self.idx] * (self.high - self.low)
        X[:, self.idx] = (np.log(phys) - self.lo) / (self.hi - self.lo)
        return X

    def fit(self, X, y):
        self.m.fit(self.g(X), y)
        return self

    def predict(self, Xs, **kw):
        return self.m.predict(self.g(Xs), **kw)

    def sample(self, Xs, n, rng):
        return self.m.sample(self.g(Xs), n, rng)

    def condition_on(self, x, y):
        w = LogInputs.__new__(LogInputs)
        w.__dict__.update(self.__dict__)
        w.m = self.m.condition_on(self.g(x)[0], y)
        return w


def initial_design(n, dim, seed):
    return qmc.LatinHypercube(dim, seed=seed).random(n)


def _recommend(problem, X, y_obs, y_true, c, model=None):
    """Noise-free value of the point the method would recommend now.  With a constraint,
    only experiments that met the spec can be recommended; if none has yet, the method is
    scored as 'no better than a typical point' (f_ref)."""
    ok = np.ones(len(y_obs), bool) if c is None else np.asarray(c) >= 0
    if not ok.any():
        return problem.f_ref
    idx = np.nonzero(ok)[0]
    if problem.noise_sd == 0 or model is None:
        return y_true[idx[int(np.argmax(np.asarray(y_obs)[idx]))]]
    mu = model.predict(np.asarray(X)[idx])[0]
    return y_true[idx[int(np.argmax(mu))]]


def run(name, problem, budget=50, seed=0, n_init=N_INIT, verbose=False):
    spec = STRATEGIES[name]
    kind = spec["kind"]
    rng = np.random.default_rng(seed)
    dim = problem.dim
    X, y_obs, y_true, rec, t_model = [], [], [], [], []
    constrained = getattr(problem, "constraint", None) is not None
    c = [] if constrained else None

    def evaluate(x, model=None):
        yo, yt = problem(x)
        X.append(np.asarray(x, float)); y_obs.append(yo); y_true.append(yt)
        if constrained:
            c.append(problem.con(x))
        rec.append(_recommend(problem, X, y_obs, y_true, c, model))

    t0 = time.time()
    if kind == "random":
        for x in rng.random((budget, dim)):
            evaluate(x)
    elif kind == "lhs":
        for x in initial_design(budget, dim, seed + 777):
            evaluate(x)
    elif kind == "simplex":
        _run_simplex(problem, budget, dim, rng, evaluate, y_obs)
    elif kind == "bo_v1":
        _run_bo_v1(problem, budget, dim, seed, evaluate, X, y_obs)
    else:
        q = spec.get("q", 1)
        for x in initial_design(n_init, dim, seed):
            evaluate(x)
        if spec["model"] in ("gp", "gpe"):
            model = GP(seed=seed) if spec["model"] == "gp" else GPEnsemble(seed=seed)
            if spec.get("warp_in"):
                from .problem import LOW, HIGH
                model = LogInputs(model, spec["warp_in"], LOW, HIGH)
        tr = TrustRegion(dim) if spec.get("tr") else None
        while len(X) < budget:
            ts = time.time()
            if spec["model"] == "nn":
                model = NNEnsemble(seed=seed * 1000 + len(X))
            y_fit = copula_transform(y_obs) if spec.get("warp", True) else np.array(y_obs)
            model.fit(np.array(X), y_fit)
            batch = []
            m = model
            if spec["acq"] == "cei" and constrained:
                model_c = GP(seed=seed + 1).fit(np.array(X), np.array(c))
                batch = [propose_constrained(model, model_c, np.array(X), y_fit,
                                             np.array(c) >= 0, rng)]
                q = 0
            box = None
            if tr is not None:                     # search only inside the trust region
                centre = np.array(X)[int(np.argmax(model.predict(np.array(X))[0]))]
                box = tr.box(centre, _length_scales(model))
            for j in range(min(q, budget - len(X))):
                acq = "ei" if spec["acq"] == "cei" else spec["acq"]   # no spec on this problem: plain EI
                x = propose(m, np.array(X), y_fit, acq, rng, t=len(X) + 1, box=box)
                batch.append(x)
                if q > 1:     # kriging believer: pretend the GP mean was observed
                    m = m.condition_on(x, m.predict(x)[0][0])
            t_model.append(time.time() - ts)
            inc_before = int(np.argmax(model.predict(np.array(X))[0]))
            for x in batch:
                evaluate(x, model)
            if tr is not None:
                # success = a new point becomes the posterior-mean incumbent after the refit
                # (noise-robust, and the same rule the benchmark uses to recommend a point)
                y_fit = copula_transform(y_obs) if spec.get("warp", True) else np.array(y_obs)
                model.fit(np.array(X), y_fit)
                inc_after = int(np.argmax(model.predict(np.array(X))[0]))
                tr.update(inc_after != inc_before and inc_after >= len(X) - len(batch))
            if verbose:
                print(f"{name} eval {len(X):3d}  best true so far {max(y_true):.4f}")
        problem.final_model = model
    return dict(strategy=name, seed=seed, problem=problem.name, noise_sd=problem.noise_sd,
                X=np.array(X), y_obs=np.array(y_obs), y_true=np.array(y_true),
                c=np.array(c if constrained else []),
                rec=np.array(rec), f_opt=problem.f_opt, f_ref=problem.f_ref,
                time_total=time.time() - t0, time_model=float(np.sum(t_model)))


def _run_simplex(problem, budget, dim, rng, evaluate, y_obs):
    """Nelder-Mead on the scaled box (clipped), stopped exactly at the budget.
    Restarts from a random point if it converges early."""
    class Stop(Exception):
        pass

    def obj(x):
        if len(y_obs) >= budget:
            raise Stop
        evaluate(np.clip(x, 0, 1))
        return -y_obs[-1]

    x0 = np.full(dim, 0.5)
    try:
        while len(y_obs) < budget:
            minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxfev=budget, initial_simplex=None, xatol=1e-4, fatol=1e-6))
            x0 = rng.random(dim)
    except Stop:
        pass


def _run_bo_v1(problem, budget, dim, seed, evaluate, X, y_obs, n_init=N_INIT):
    """Faithful copy of bo_5knobs.py (random start, RBF length 0.3, EI over 5,000 random
    candidates, incumbent = best standardised observation)."""
    rs = np.random.RandomState(seed)
    for x in rs.rand(n_init, dim):
        evaluate(x)
    gp = GP(kernel="rbf", ard=False, fixed=dict(ls=0.3, s2=1.0, noise=1e-6))
    while len(X) < budget:
        Xa, Ya = np.array(X), np.array(y_obs)
        gp.fit(Xa, Ya)
        cand = rs.rand(5000, dim)
        mu, sd = gp.predict(cand)
        ei = expected_improvement(mu, sd, Ya.max())
        evaluate(cand[np.argmax(ei)])
