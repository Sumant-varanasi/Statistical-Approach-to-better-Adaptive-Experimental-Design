"""
GP vs neural-network surrogate: how well does each one PREDICT the digital twin?

This is the controlled comparison asked for in the project plan (same data, same setup).
For each objective, training-set size n and repeat:
    * draw n training experiments (Latin hypercube) and run the twin on them
    * fit three surrogates on the same data:
          GP v1      RBF kernel, length-scale fixed at 0.3 (the original bo_5knobs.py GP)
          GP         Matern-5/2 ARD, all hyper-parameters fitted by marginal likelihood
          NN         deep ensemble of 5 small networks (bo/nn.py)
    * score them on two test sets:
          'whole box'    1,000 uniform random points
          'near optimum' 500 points scattered around the true optimum (what BO cares about)

Metrics
    NRMSE      root-mean-square error / standard deviation of the test values (0 = perfect, 1 = useless)
    coverage   fraction of test values inside the model's 95 % interval (ideal = 0.95;
               below = over-confident, above = too cautious)
    NLPD       negative log predictive density: rewards accurate AND honest uncertainty (lower = better)
    fit time   seconds to train the model once

Outputs:  results/surrogate_study.csv,  figures/surrogate_study.png
Run:      python surrogate_study.py            (about 10 minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
import time
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bo.problem import Problem
from bo.gp import GP
from bo.nn import NNEnsemble
from bo.loop import initial_design

SIZES = [10, 20, 40, 80, 160]
REPEATS = 5
MODELS = {
    "GP v1 (fixed RBF)": lambda s: GP(kernel="rbf", ard=False, fixed=dict(ls=0.3, s2=1.0, noise=1e-6)),
    "GP (fitted Matern ARD)": lambda s: GP(seed=s),
    "NN ensemble": lambda s: NNEnsemble(seed=s),
}
COLORS = {"GP v1 (fixed RBF)": "#1baf7a", "GP (fitted Matern ARD)": "#2a78d6", "NN ensemble": "#eda100"}


def test_sets(prob, rng):
    Xu = rng.random((1000, prob.dim))
    Xn = np.clip(prob.x_opt + rng.normal(0, 0.1, (500, prob.dim)), 0, 1)
    return {"whole box": (Xu, np.array([prob.true(x) for x in Xu])),
            "near optimum": (Xn, np.array([prob.true(x) for x in Xn]))}


def score(mu, sd, y):
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return dict(nrmse=float(np.sqrt(np.mean((mu - y) ** 2)) / y.std()),
                coverage=float(np.mean(np.abs(z) <= 1.96)),
                nlpd=float(np.mean(-norm.logpdf(y, mu, sd))))


def main():
    rows = []
    for obj in ["yield", "margin"]:
        prob = Problem(obj)
        tests = test_sets(prob, np.random.default_rng(42))
        for n in SIZES:
            for rep in range(REPEATS):
                X = initial_design(n, prob.dim, seed=1000 * n + rep)
                y = np.array([prob.true(x) for x in X])
                for name, make in MODELS.items():
                    m = make(rep)
                    t = time.time()
                    m.fit(X, y)
                    ft = time.time() - t
                    for tname, (Xt, yt) in tests.items():
                        mu, sd = m.predict(Xt)
                        rows.append(dict(objective=obj, n_train=n, repeat=rep, model=name,
                                         test_set=tname, fit_time=ft, **score(mu, sd, yt)))
                print(f"{obj:6s} n={n:3d} repeat {rep}  done", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/surrogate_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, "figures/surrogate_study.png")
    for obj in ["yield", "margin"]:
        print(f"\n{obj}: median over repeats at n = 40 / 160")
        for name in MODELS:
            for tname in ["whole box", "near optimum"]:
                vals = []
                for n in (40, 160):
                    r = [x for x in rows if x["objective"] == obj and x["model"] == name
                         and x["test_set"] == tname and x["n_train"] == n]
                    vals.append(f"NRMSE {np.median([x['nrmse'] for x in r]):.3f} cov {np.median([x['coverage'] for x in r]):.2f}")
                print(f"  {name:24s} {tname:13s} " + "   |   ".join(vals))


def plot(rows, path):
    metrics = [("nrmse", "Prediction error, NRMSE (lower is better)", True),
               ("coverage", "95 % interval coverage (ideal 0.95)", False),
               ("nlpd", "NLPD (lower is better)", False)]
    panels = [(o, t) for o in ["yield", "margin"] for t in ["whole box", "near optimum"]]
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(14, 9.5), sharex=True)
    for j, (obj, tname) in enumerate(panels):
        for i, (key, lab, logy) in enumerate(metrics):
            ax = axes[i, j]
            for name in MODELS:
                med, lo, hi = [], [], []
                for n in SIZES:
                    v = [r[key] for r in rows if r["objective"] == obj and r["test_set"] == tname
                         and r["model"] == name and r["n_train"] == n]
                    med.append(np.median(v)); lo.append(np.min(v)); hi.append(np.max(v))
                ax.fill_between(SIZES, lo, hi, color=COLORS[name], alpha=0.12, lw=0)
                ax.plot(SIZES, med, "-o", color=COLORS[name], lw=2, ms=4, label=name)
            if key == "coverage":
                ax.axhline(0.95, color="#898781", lw=0.8, ls=":")
                ax.set_ylim(0, 1.02)
            if logy:
                ax.set_yscale("log")
            if key == "nlpd":
                ax.set_yscale("symlog")
            ax.set_xscale("log")
            ax.set_xticks(SIZES)
            ax.set_xticklabels(SIZES)
            if i == 0:
                ax.set_title(f"{'Yield' if obj == 'yield' else 'Margin'}: {tname}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("Training experiments", fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.grid(color="#e1e0d9", lw=0.6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Surrogate accuracy vs amount of data (line = median of 5 repeats, band = min-max)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
