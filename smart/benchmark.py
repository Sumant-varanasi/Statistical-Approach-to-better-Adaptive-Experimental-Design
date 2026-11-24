"""
Head-to-head benchmark on the virtual reactors, run the way the real reactor is run: catalyst
fixed, 5 operator screening runs first, then every method proposes runs within the step limits.

Settings (scenarios)
    simple         the project's A+B->C, A->D network, stable catalyst
    complex        A+B<=>C, C+B->E, A->D, stable catalyst
    complex_deact  the complex network on a catalyst that loses activity on stream (coking:
                   faster when hotter and with more A); the protocol repeats the base case every
                   5th run as a check run, so drift can be seen at all
Methods
    smart_v7     the v7 "hybrid" advisor: v6 + a stand-alone neural network that predicts the profit
                 from the knobs alone; the advisor trusts whichever has predicted unseen runs better
    smart_v6     the v6 advisor: v5 + a committee correction (none / GP / neural-network ensemble,
                 weighted by held-out prediction; smart/committee.py)
    smart        the v5 advisor: kinetics ladder (inhibition, its heat, decay) + horizon planner
    smart_v4     the v4 advisor: plain kinetics + one-step Expected Improvement with the purposeful step
    chemist      a scripted rules-of-thumb policy with momentum (advisor.ChemistPolicy)
    plain_bo     black-box BO (v2's fitted GP + EI) under the same step limits, profit only
    nn_only      a neural network alone (deep ensemble + EI) under the same step limits, profit only
    random_walk  a random allowed step each run
Ablations (what does each part of the committee contribute?)
    smart_v6_nonn  v6 without its neural networks (members: none + GP)
    smart_v6_nogp  v6 without its Gaussian process (members: none + neural networks)
Stress test (complex only)
    smart_missing_E     the v5 advisor with the over-reaction C + B -> E left out of its model
    smart_v6_missing_E, smart_v7_missing_E, smart_v6_nonn_missing_E, smart_v6_nogp_missing_E
                        the same for v6, v7 and the two ablations
To add a method: put its label in METHODS (or STRESS) and, if it is a smart-advisor variant, its
run_smart keyword arguments in SMART_KWARGS; a baseline policy goes in advisor.POLICIES.

Every method sees the same virtual reactor, the same GC noise and the SAME 5 screening runs
for a given seed; profit is always computed from the noisy GC reading.  Each campaign is
saved to results/smart_bench/<setting>/<method>_s<seed>_b<budget>.npz, so the benchmark can be
stopped and resumed, and campaigns of different budgets never get mixed up.  Saved per campaign:
X, gc, y_obs, y_true, rec (true profit of the recommended settings at the current activity),
cum_true (profit earned so far), activity, check (which runs were check runs), time_total.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
import warnings
import numpy as np

OUT = os.environ.get("SMART_BENCH_DIR", os.path.join("results", "smart_bench"))
SETTINGS = {
    "simple": dict(system="simple", deactivation=0.0, check_every=0),
    "complex": dict(system="complex", deactivation=0.0, check_every=0),
    "complex_deact": dict(system="complex", deactivation=0.02 / 24.0, check_every=5),
}
METHODS = {
    "smart_v6": "Smart advisor v6 (v5 + committee correction: GP + neural networks)",
    "smart": "Smart advisor v5 (kinetics ladder + horizon planner)",
    "smart_v4": "Smart advisor v4 (plain kinetics + one-step EI)",
    "chemist": "Scripted rules-of-thumb policy",
    "plain_bo": "Plain BO (GP + EI, profit only)",
    "random_walk": "Random walk",
    # ---- v7 and what it is compared with
    "smart_v7": "Smart advisor v7 (hybrid: v6 + a stand-alone neural network, trusting whichever predicts unseen runs better)",
    "nn_only": "Neural network alone (no physics, profit only)",
    "smart_v6_nonn": "Smart advisor v6 without its neural networks (physics + GP correction)",
    "smart_v6_nogp": "Smart advisor v6 without its Gaussian process (physics + neural-network correction)",
}
# stress test, complex system only: advisor whose model is missing the over-reaction
STRESS = {"smart_missing_E": "Smart advisor v5, over-reaction C+B->E missing from its model",
          "smart_v6_missing_E": "Smart advisor v6, over-reaction C+B->E missing from its model",
          "smart_v7_missing_E": "Smart advisor v7 (hybrid), over-reaction C+B->E missing from its model",
          "smart_v6_nonn_missing_E": "Smart advisor v6 without its neural networks, over-reaction C+B->E missing from its model",
          "smart_v6_nogp_missing_E": "Smart advisor v6 without its Gaussian process, over-reaction C+B->E missing from its model"}
LABELS = {**METHODS, **STRESS}
# the smart-advisor variants: method -> keyword arguments of advisor.run_smart.  Any method that
# is not listed here is a baseline policy (advisor.POLICIES) and goes to advisor.run_baseline.
_MISSING_E = dict(model="complex_missing_E")
_NONN, _NOGP = dict(correction="committee", members=("none", "gp")), dict(correction="committee", members=("none", "nn"))
SMART_KWARGS = {
    "smart": {},
    "smart_v4": dict(planner="greedy", models=("plain",)),
    "smart_missing_E": _MISSING_E,
    "smart_v6": dict(correction="committee"),
    "smart_v6_missing_E": dict(correction="committee", **_MISSING_E),
    "smart_v7": dict(correction="hybrid"),
    "smart_v7_missing_E": dict(correction="hybrid", **_MISSING_E),
    "smart_v6_nonn": _NONN,
    "smart_v6_nonn_missing_E": dict(**_NONN, **_MISSING_E),
    "smart_v6_nogp": _NOGP,
    "smart_v6_nogp_missing_E": dict(**_NOGP, **_MISSING_E),
}
SETTING_LABEL = {"simple": "Simple reaction  (A+B->C, A->D)",
                 "complex": "Complex reaction  (A+B<=>C, C+B->E, A->D)",
                 "complex_deact": "Complex reaction, catalyst losing activity (check run every 5th)"}
SYSTEM_LABEL = SETTING_LABEL          # backwards compatibility
N_SCREEN = 5


def applies(method, setting):
    return method not in STRESS or setting != "simple"


def path_for(setting, method, seed, budget=25):
    return os.path.join(OUT, setting, f"{method}_s{seed}_b{budget}.npz")


def run_one(setting, method, seed, budget):
    from smart import advisor
    cfg = SETTINGS[setting]
    common = dict(budget=budget, seed=seed, deactivation=cfg["deactivation"], check_every=cfg["check_every"])
    if method in SMART_KWARGS:
        return advisor.run_smart(cfg["system"], **SMART_KWARGS[method], **common)
    return advisor.run_baseline(cfg["system"], method, **common)


def job(args):
    setting, method, seed, budget = args
    p = path_for(setting, method, seed, budget)
    if os.path.exists(p):
        return args, None
    warnings.filterwarnings("ignore")
    t0 = time.time()
    h = run_one(setting, method, seed, budget)
    h = {k: v for k, v in h.items() if isinstance(v, (np.ndarray, float, int, str, np.floating))}
    h["time_total"] = time.time() - t0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **h)
    return args, (float(h["rec"][-1]), float(h["cum_true"][-1]), h["time_total"])


def run_all(settings=tuple(SETTINGS), methods=tuple(LABELS), seeds=10, budget=25, workers=None, verbose=True):
    from multiprocessing import Pool
    jobs = [(st, m, k, budget) for k in range(seeds) for st in settings for m in methods if applies(m, st)]
    todo = [j for j in jobs if not os.path.exists(path_for(*j))]
    if not todo:
        return
    workers = workers or max(1, min(len(todo), (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"Running {len(todo)} campaigns on {workers} processes "
              f"(finished ones are cached in {OUT}) ...", flush=True)
    t0 = time.time()
    # longest jobs (smart advisors) first so the pool finishes evenly
    todo.sort(key=lambda j: (not j[1].startswith("smart"), j[0] == "simple"))
    with Pool(workers) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo), 1):
            if verbose and r is not None:
                print(f"  [{i:3d}/{len(todo)} {time.time() - t0:5.0f}s] {a[0]:14s} {a[1]:16s} seed {a[2]:2d}"
                      f"  final {r[0]:7.2f}  earned {r[1]:7.1f}  ({r[2]:.0f}s)", flush=True)


def load(setting, method, seeds, budget=25):
    out = []
    for k in range(seeds):
        p = path_for(setting, method, k, budget)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            out.append({key: d[key] for key in d.files})
    return out
