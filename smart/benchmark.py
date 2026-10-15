"""
Head-to-head benchmark on the virtual reactors (both reaction systems), run the way the real
reactor is run: catalyst fixed, 5 operator screening runs first, then every method proposes
runs within the step limits.

Methods
    smart        the physics-informed advisor (smart/advisor.py)
    plain_bo     black-box BO (v2's fitted GP + EI) under the same step limits, profit only
    chemist      an experienced operator's rules of thumb + momentum (advisor.ChemistPolicy)
    random_walk  a random allowed step each run
Stress test (complex system only)
    smart_missing_E   the advisor with the over-reaction C + B -> E left out of its model

Every method sees the same virtual reactor, the same GC noise and the SAME 5 screening runs
for a given seed; profit is always computed from the noisy GC reading.  Each campaign is
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
    "plain_bo": "Plain BO (GP + EI, profit only)",
    "chemist": "Chemist's rules of thumb",
    "random_walk": "Random walk",
}
# stress test, complex system only: advisor whose model is missing the over-reaction
STRESS = {"smart_missing_E": "Smart advisor, over-reaction C+B->E missing from its model"}
LABELS = {**METHODS, **STRESS}
SYSTEM_LABEL = {"simple": "Simple reaction  (A+B->C, A->D)",
                "complex": "Complex reaction  (A+B<=>C, C+B->E, A->D)"}
N_SCREEN = 5


def path_for(system, method, seed):
    return os.path.join(OUT, system, f"{method}_s{seed}.npz")


def job(args):
    system, method, seed, budget = args
    p = path_for(system, method, seed)
    if os.path.exists(p):
        return args, None
    warnings.filterwarnings("ignore")
    from smart import advisor
    t0 = time.time()
    if method.startswith("smart"):
        model = "complex_missing_E" if method == "smart_missing_E" else None
        h = advisor.run_smart(system, budget=budget, seed=seed, model=model)
    else:
        h = advisor.run_baseline(system, method, budget=budget, seed=seed)
    h = {k: v for k, v in h.items() if isinstance(v, (np.ndarray, float, int, str, np.floating))}
    h["time_total"] = time.time() - t0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **h)
    return args, (float(h["rec"][-1]), h["time_total"])


def run_all(systems=("simple", "complex"), methods=tuple(LABELS), seeds=10, budget=25,
            workers=None, verbose=True):
    from multiprocessing import Pool
    jobs = [(s, m, k, budget) for k in range(seeds) for s in systems for m in methods
            if m not in STRESS or s == "complex"]
    todo = [j for j in jobs if not os.path.exists(path_for(*j[:3]))]
    if not todo:
        return
    workers = workers or max(1, min(len(todo), (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"Running {len(todo)} campaigns on {workers} processes "
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
