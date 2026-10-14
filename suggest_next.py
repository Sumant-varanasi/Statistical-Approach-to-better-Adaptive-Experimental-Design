"""
Next-experiment recommender: the human-in-the-loop tool.

Give it a CSV of experiments already done (on the twin OR on the real reactor) and it
recommends the next batch of operating points to try, with the GP's predicted result and
its uncertainty for each.  A person reviews the list and decides what actually gets run.

CSV format: one row per experiment, columns
    P, V_A, V_B, T, Wcat, <result column>          (physical units: bar, mL/min, mL/min, K, g)

Examples
    python suggest_next.py --demo                       # makes demo_experiments.csv from the twin, then suggests
    python suggest_next.py experiments.csv --y Y_C --batch 4
    python suggest_next.py experiments.csv --y margin --acq ucb --batch 3
    python suggest_next.py experiments.csv --y Y_C --fix T=600 --fix P=10   # hold knobs fixed

What it prints
    * what the GP learned: how sensitive the result is to each knob (from the fitted
      length-scales: short length-scale = the result changes quickly with that knob)
    * the best experiment so far and where the GP thinks the optimum is
    * the recommended batch, in physical units, with predicted mean and 95 % range
The recommendations are also written to suggestions.csv.

Safety: this only SUGGESTS.  Any physical experiment still needs expert review, a chemical
safety check and supervisor approval (project plan, section 11).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import numpy as np
from bo.problem import KEYS, NAMES, LOW, HIGH, unscale, scale, Problem
from bo.gp import GP
from bo.loop import copula_transform, initial_design, LogInputs
from bo.acquisition import propose


def read_csv(path, ycol):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    missing = [k for k in KEYS + [ycol] if k not in rows[0]]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; expected {KEYS + [ycol]}")
    phys = np.array([[float(r[k]) for k in KEYS] for r in rows])
    y = np.array([float(r[ycol]) for r in rows])
    return phys, y


def make_demo(path, n=12, seed=0):
    """A small starting data set from the twin, so the tool can be tried immediately."""
    prob = Problem("yield")
    X = initial_design(n, 5, seed)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(KEYS + ["Y_C"])
        for x in X:
            w.writerow([*np.round(unscale(x), 3), round(prob.true(x), 5)])
    print(f"Wrote {path} with {n} demo experiments from the digital twin.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="demo_experiments.csv")
    ap.add_argument("--y", default="Y_C", help="name of the result column to MAXIMISE")
    ap.add_argument("--minimise", action="store_true", help="minimise the result instead")
    ap.add_argument("--batch", type=int, default=3, help="how many experiments to suggest")
    ap.add_argument("--acq", default="ei", choices=["ei", "ucb", "pi", "ts"])
    ap.add_argument("--fix", action="append", default=[], help="hold a knob fixed, e.g. --fix T=600")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="create demo data from the twin first")
    a = ap.parse_args()

    if a.demo:
        make_demo(a.csv)
    phys, y = read_csv(a.csv, a.y)
    sign = -1.0 if a.minimise else 1.0
    X = scale(phys)
    if (X < -1e-9).any() or (X > 1 + 1e-9).any():
        print("WARNING: some experiments are outside the search box used by the optimiser:")
        for k, lo, hi in zip(KEYS, LOW, HIGH):
            print(f"   {k}: {lo} to {hi}")
    fixed = {}
    for f in a.fix:
        k, v = f.split("=")
        i = KEYS.index(k)
        fixed[i] = (float(v) - LOW[i]) / (HIGH[i] - LOW[i])

    rng = np.random.default_rng(a.seed)
    yt = copula_transform(sign * y)          # robust to outliers; order is all BO needs
    # the GP sees pressure and catalyst mass on a log scale (best setting in the benchmark)
    gp = LogInputs(GP(seed=a.seed), (0, 4), LOW, HIGH).fit(X, yt)
    gp_raw = LogInputs(GP(seed=a.seed), (0, 4), LOW, HIGH).fit(X, y)   # real units, for readable predictions

    print(f"{len(y)} experiments read from {a.csv}.  Objective: {'minimise' if a.minimise else 'maximise'} {a.y}\n")
    print("How sensitive the result is to each knob (from the fitted GP; 1.00 = most sensitive;")
    print("P and Wcat are judged per factor-of-ten change, the others per unit of range):")
    sens = (1 / gp_raw.m.ls) / (1 / gp_raw.m.ls).max()
    for name, s in sorted(zip(NAMES, sens), key=lambda t: -t[1]):
        print(f"   {name:14s} {s:5.2f}  {'#' * int(round(20 * s))}")
    print(f"   (estimated measurement noise sd = {np.sqrt(gp_raw.m.sn2) * gp_raw.m.y_std:.3g} in {a.y} units)\n")

    ib = int(np.argmax(sign * y))
    print("Best experiment so far: " + ", ".join(f"{k}={v:.3g}" for k, v in zip(KEYS, phys[ib])) +
          f"  ->  {a.y} = {y[ib]:.4g}\n")

    batch, m = [], gp
    for j in range(a.batch):
        x = propose(m, X, yt, a.acq, rng, t=len(y) + 1)
        for i, v in fixed.items():
            x[i] = v
        batch.append(x)
        m = m.condition_on(x, m.predict(x)[0][0])       # kriging believer for the batch

    mu, sd = gp_raw.predict(np.array(batch))
    print(f"Suggested next {a.batch} experiment(s)  (acquisition = {a.acq.upper()}):")
    hdr = "".join(f"{k:>9s}" for k in KEYS) + f"{'predicted ' + a.y:>22s}{'95% range':>22s}"
    print("   #" + hdr)
    out = []
    for j, (x, m_, s_) in enumerate(zip(batch, mu, sd), 1):
        p = unscale(x)
        lo, hi = m_ - 1.96 * s_, m_ + 1.96 * s_
        print(f"  {j:2d}" + "".join(f"{v:9.3g}" for v in p) + f"{m_:22.4g}{f'[{lo:.3g}, {hi:.3g}]':>22s}")
        out.append([j, *np.round(p, 3), round(m_, 5), round(lo, 5), round(hi, 5)])
    with open("suggestions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suggestion"] + KEYS + [f"predicted_{a.y}", "lower_95", "upper_95"])
        w.writerows(out)
    if a.y == "Y_C" and np.any(mu + 1.96 * sd > 1):
        print("\nNote: the GP does not know that yield cannot exceed 1, so ranges above 1.0 just mean"
              "\n'very uncertain, possibly near the maximum'.")
    print("\nSaved suggestions.csv.  Review before running: these are suggestions, not instructions.")


if __name__ == "__main__":
    main()
