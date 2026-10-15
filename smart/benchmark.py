"""
Head-to-head benchmark on the virtual reactors (both reaction systems).

Methods
    smart       the physics-informed advisor (smart/advisor.py)
    gp_ei_logw  best plain BO from the v2 benchmark (fitted GP + EI, log P & W_cat), profit only
    ofat        trial and error, one factor at a time (what is usually done by hand)
    random      random search
Stress test (complex system only)
    smart_missing_E   the advisor with the over-reaction C + B -> E left out of its model

Every method sees the same virtual reactor with the same GC noise; profit is always computed
from the noisy GC reading.  Methods with the same seed start from the same Latin-hypercube
design (OFAT starts from the project's base case instead, as a person would).  Each run is
saved to results/smart_bench/<system>/<method>_s<seed>.npz, so the benchmark can be stopped
and resumed.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
import warnings
import numpy as np

OUT = os.environ.get("SMART_BENCH_DIR", os.path.join("results", "smart_bench"))
METHODS = {
    "smart": "Smart advisor (physics + GP)",
    "gp_ei_logw": "Plain BO (GP + EI, v2 best)",
    "ofat": "Trial and error (one factor at a time)",
    "random": "Random search",
}
# stress test, complex system only: advisor whose model is missing the over-reaction
STRESS = {"smart_missing_E": "Smart advisor, over-reaction C+B->E missing from its model"}
LABELS = {**METHODS, **STRESS}
SYSTEM_LABEL = {"simple": "Simple reaction  (A+B->C, A->D)",
                "complex": "Complex reaction  (A+B<=>C, C+B->E, A->D)"}


def path_for(system, method, seed):
    return os.path.join(OUT, system, f"{method}_s{seed}.npz")


def job(args):
    system, method, seed, budget, n_init = args
    p = path_for(system, method, seed)
    if os.path.exists(p):
        return args, None
    warnings.filterwarnings("ignore")
    from smart import advisor
    t0 = time.time()
    if method.startswith("smart"):
        model = "complex_missing_E" if method == "smart_missing_E" else None
        h = advisor.run_smart(system, budget=budget, seed=seed, n_init=n_init, model=model)
        h.pop("info", None)
        h.pop("advisor", None)
    elif method == "ofat":
        h = advisor.run_ofat(system, budget=budget, seed=seed)
    else:
        from bo import loop
        h = loop.run(method, advisor.ProfitProblem(system, seed), budget=budget, seed=seed,
                     n_init=n_init)
    h = {k: v for k, v in h.items() if isinstance(v, (np.ndarray, float, int, str, np.floating))}
    h["time_total"] = time.time() - t0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **h)
    return args, (float(h["rec"][-1]), h["time_total"])


def run_all(systems=("simple", "complex"), methods=tuple(LABELS), seeds=10, budget=30,
            n_init=8, workers=None, verbose=True):
    from multiprocessing import Pool
    jobs = [(s, m, k, budget, n_init) for k in range(seeds) for s in systems for m in methods
            if m not in STRESS or s == "complex"]
    todo = [j for j in jobs if not os.path.exists(path_for(*j[:3]))]
    if not todo:
        return
    workers = workers or max(1, min(len(todo), (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"Running {len(todo)} optimisation campaigns on {workers} processes "
              f"(finished ones are cached in {OUT}) ...", flush=True)
    t0 = time.time()
    # longest jobs (smart, complex) first so the pool finishes evenly
    todo.sort(key=lambda j: (not j[1].startswith("smart"), j[0] != "complex"))
    with Pool(workers) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo), 1):
            if verbose and r is not None:
                print(f"  [{i:3d}/{len(todo)} {time.time() - t0:5.0f}s] {a[0]:8s} {a[1]:16s} seed {a[2]:2d}"
                      f"  final profit {r[0]:7.2f}  ({r[1]:.0f}s)", flush=True)


def load(system, method, seeds):
    out = []
    for k in range(seeds):
        p = path_for(system, method, k)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            out.append({key: d[key] for key in d.files})
    return out
