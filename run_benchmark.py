"""
Multi-seed benchmark: every strategy x every seed x every problem setting.

Each run is saved as its own file in results/bench/<setting>/<strategy>_s<seed>.npz, so the
benchmark can be stopped and restarted (finished runs are skipped).

Run:   python run_benchmark.py                       (all settings, 20 seeds, 2 processes)
       python run_benchmark.py --seeds 5 --workers 4
       python run_benchmark.py --settings margin --strategies gp_ei random
Then:  python analyse_benchmark.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread per process: much faster
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")     # when several processes run at once
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import time
import numpy as np
from multiprocessing import Pool
from bo.problem import Problem, SPEC
from bo import loop

ALL = [s for s in loop.STRATEGIES if s != "cgp_ei"]
SETTINGS = {                      # name: (objective, noisy, product-spec constraint, strategies)
    "yield": ("yield", False, None, ALL),
    "margin": ("margin", False, None, ALL),
    "margin_noisy": ("margin", True, None, ALL),
    # profit subject to selectivity >= 0.97: constrained BO vs methods that ignore the spec
    "margin_spec": ("margin", False, SPEC, ["random", "lhs", "bo_v1", "gp_ei", "cgp_ei"]),
}
OUT = "results/bench"


def path_for(setting, strat, seed):
    return os.path.join(OUT, setting, f"{strat}_s{seed}.npz")


def job(args):
    setting, strat, seed, budget = args
    p = path_for(setting, strat, seed)
    if os.path.exists(p):
        return setting, strat, seed, None
    obj, noisy, con, _ = SETTINGS[setting]
    h = loop.run(strat, Problem(obj, noisy=noisy, seed=seed, constraint=con), budget=budget, seed=seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **{k: v for k, v in h.items()})
    return setting, strat, seed, (h["f_opt"] - h["rec"][-1], h["time_total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="default: every strategy that applies to each setting")
    a = ap.parse_args()
    # interleave seeds first so partial results are usable for every strategy early
    jobs = [(st, s, k, a.budget) for k in range(a.seeds) for st in a.settings
            for s in (a.strategies or SETTINGS[st][3]) if s in SETTINGS[st][3]]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (st, s, k, r) in enumerate(pool.imap_unordered(job, jobs), 1):
            if r is not None:
                print(f"[{i:4d}/{len(jobs)} {time.time() - t0:6.0f}s] {st:13s} {s:10s} seed {k:2d}"
                      f"  final regret {r[0]:9.4f}  ({r[1]:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
