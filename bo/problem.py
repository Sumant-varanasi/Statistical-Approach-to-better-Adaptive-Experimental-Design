"""
Optimisation problems built on the reactor digital twin.

Same 5-knob search box as bo_5knobs.py (inert flow fixed at 10 mL/min):
    P (bar), V_A (mL/min), V_B (mL/min), T (K), Wcat (g)
Every optimiser works in scaled units x in [0, 1]^5.

Two objectives (both MAXIMISED):

  'yield'   Y_C = F_C,out / F_A,0           (the original objective, dimensionless)
            Easy: about 7 % of the box is within 5 % of the best value, the optimum
            is a broad plateau at high P / high B excess / lots of catalyst.

  'margin'  a process-economics objective, in "money units per hour" (placeholder prices):
                margin = v_C * C produced
                       - c_feed * (A fed + B fed)           unreacted feed is wasted
                       - c_comp * total feed * ln(P / 1 bar)  compression energy
                       - c_cat * Wcat                        catalyst cost
            all flows in mmol/h.  Hard: the optimum is a small region INSIDE the box
            (T near 590 K, under 1 g catalyst, B slightly below A), ~90 % of the box loses money.
            Prices are made up and must be replaced by agreed values.

Measurement noise: an optional Gaussian noise of standard deviation `noise_sd` (in objective
units) is added to every observation, imitating GC measurement error.  The noise-free value
is always recorded too, so regret can be measured honestly.
"""
import json
import os
import numpy as np
from reactor_twin import simulate

NAMES = ["P (bar)", "V_A (mL/min)", "V_B (mL/min)", "T (K)", "Wcat (g)"]
KEYS = ["P", "V_A", "V_B", "T", "Wcat"]
LOW = np.array([1.0, 1.0, 1.0, 450.0, 0.05])
HIGH = np.array([100.0, 50.0, 50.0, 850.0, 5.0])
V_I = 10.0
DIM = 5

# placeholder economics (money units per mmol, per mmol*ln(bar), per gram)
PRICES = dict(v_C=1.0, c_feed=0.20, c_comp=0.03, c_cat=3.0)
MMOL_H = 3.6e6          # mol/s -> mmol/h

_OPTIMA_FILE = os.path.join(os.path.dirname(__file__), "optima.json")


def unscale(x):
    return LOW + np.asarray(x) * (HIGH - LOW)


def scale(phys):
    return (np.asarray(phys) - LOW) / (HIGH - LOW)


def twin(x):
    """Run the twin at scaled x and return the full result dict."""
    P, VA, VB, T, W = unscale(np.clip(x, 0, 1))
    return simulate(V_A=VA, V_B=VB, V_I=V_I, Wcat=W, T=T, P=P)


def yield_of_C(x):
    return twin(x)["Y_C"]


def margin(x, prices=PRICES):
    r = twin(x)
    P, W = unscale(np.clip(x, 0, 1))[[0, 4]]
    FC = r["F_out"]["C"] * MMOL_H
    feed = (r["F0"]["A"] + r["F0"]["B"]) * MMOL_H
    FT0 = r["FT_0"] * MMOL_H
    return (prices["v_C"] * FC - prices["c_feed"] * feed
            - prices["c_comp"] * FT0 * np.log(P) - prices["c_cat"] * W)


OBJECTIVES = {
    "yield": dict(fn=yield_of_C, label="Yield of C, Y_C", units="", noise_sd=0.01),
    "margin": dict(fn=margin, label="Process margin", units="money units/h", noise_sd=0.5),
}


class Problem:
    """
    prob = Problem('margin', noisy=True, seed=3)
    y_obs = prob(x)            # noisy observation (what the optimiser sees)
    prob.true(x)               # noise-free value
    prob.f_opt, prob.x_opt     # global optimum (pre-computed, cached in optima.json)

    Optional product-spec constraint, e.g. constraint=("S_C", 0.97) means selectivity to C
    must be at least 0.97.  prob.con(x) returns S_C - 0.97 (>= 0 means the spec is met), and
    f_opt is then the best value AMONG operating points that meet the spec.
    """

    def __init__(self, name, noisy=False, seed=0, noise_sd=None, constraint=None):
        spec = OBJECTIVES[name]
        self.name = name
        self.fn = spec["fn"]
        self.label = spec["label"]
        self.units = spec["units"]
        self.noise_sd = (spec["noise_sd"] if noise_sd is None else noise_sd) if noisy else 0.0
        self.rng = np.random.default_rng(10_000 + seed)
        self.dim = DIM
        self.constraint = constraint
        opt = global_optimum(name, constraint=constraint)
        self.f_opt, self.x_opt = opt["f_opt"], np.array(opt["x_opt"])
        self.f_ref = opt["f_median"]      # median over the box, used to normalise regret
        self.n_calls = 0

    def true(self, x):
        return float(self.fn(x))

    def con(self, x):
        """Constraint margin: >= 0 means the operating point meets the spec."""
        key, limit = self.constraint
        return float(twin(x)[key] - limit)

    def __call__(self, x):
        self.n_calls += 1
        f = self.true(x)
        return f + (self.rng.normal(0, self.noise_sd) if self.noise_sd > 0 else 0.0), f


def global_optimum(name, recompute=False, constraint=None):
    """Global optimum of an objective via differential evolution + several polished
    restarts.  Cached in bo/optima.json because it takes ~1 minute.  With a constraint the
    spec is enforced by a large penalty, and the result is checked to be feasible."""
    key = name if constraint is None else f"{name}|{constraint[0]}>={constraint[1]}"
    cache = {}
    if os.path.exists(_OPTIMA_FILE):
        with open(_OPTIMA_FILE) as fh:
            cache = json.load(fh)
    if key in cache and not recompute:
        return cache[key]
    from scipy.optimize import differential_evolution
    base = OBJECTIVES[name]["fn"]
    if constraint is None:
        fn = base
    else:
        ck, cl = constraint
        fn = lambda x: base(x) - 1e4 * max(0.0, cl - twin(x)[ck])
    best = None
    for s in range(3):
        res = differential_evolution(lambda x: -fn(x), [(0, 1)] * DIM, seed=s, tol=1e-12,
                                     maxiter=400, popsize=20, polish=True)
        if best is None or res.fun < best.fun:
            best = res
    rng = np.random.default_rng(123)
    ys = np.array([base(x) for x in rng.random((4000, DIM))])
    cache[key] = dict(f_opt=float(-best.fun), x_opt=[float(v) for v in best.x],
                       x_opt_phys=[float(v) for v in unscale(best.x)],
                       f_median=float(np.median(ys)),
                       frac_within_5pct=float(np.mean(ys >= -best.fun - 0.05 * (-best.fun - np.median(ys)))))
    tmp = _OPTIMA_FILE + ".tmp"            # atomic write: other processes may be reading it
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp, _OPTIMA_FILE)
    return cache[key]


SPEC = ("S_C", 0.97)        # product spec used for the constrained benchmark

if __name__ == "__main__":
    for nm in OBJECTIVES:
        o = global_optimum(nm, recompute=True)
        print(nm, json.dumps(o, indent=1))
    print("margin with spec", json.dumps(global_optimum("margin", True, SPEC), indent=1))
