"""
Brute-force reference campaign on the reactor digital twin (the "simulate a lot" side).

Runs N random experiments over the SAME 5-knob box that bo_5knobs.py optimises and
records the best yield found as the campaign grows.  Also draws the two reference
figures for the digital-twin side of the comparison.

Outputs
    results/simulation_history.csv    one row per simulated experiment
    figures/simulation_progress.png   best yield so far vs number of simulations
    figures/twin_5param_maps.png      all 5 knobs at once: a yield heat map for every pair
                                      of knobs (slices through the best simulated point)
                                      with 1-D slices on the diagonal; the brute-force best
                                      and the BO best are both marked
    figures/sensitivity_bar.png       how strongly each knob moves the yield (for reference)

Run:   python simulation_campaign.py                 (defaults: --n 20000 --seed 1 --grid 30)
       python simulation_campaign.py --n 5000        (quicker)
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reactor_twin import simulate
from bo_5knobs import NAMES, LOW, HIGH, V_I, unscale, f, COL_BO

COL_SIM, COL_SIM_LIGHT = "#eb6834", "#f6b89f"       # simulation colours, reused by compare.py
BASE = dict(P=10.0, V_A=20.0, V_B=20.0, T=600.0, Wcat=1.0)   # base case used by sweeps.py
KEYS = ["P", "V_A", "V_B", "T", "Wcat"]              # knob order everywhere


def run_campaign(n, seed):
    """Simulate n random experiments.  Returns X (n x 5 scaled) and Y (n yields)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    Y = np.empty(n)
    for i, x in enumerate(X):
        Y[i] = f(x)
        if (i + 1) % 2000 == 0:
            print(f"  simulated {i + 1:6d} / {n}   best so far = {Y[:i + 1].max():.4f}")
    return X, Y


def save_history(X, Y, path):
    """Same column layout as results/bo_history.csv so compare.py can read both."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best = np.maximum.accumulate(Y)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "phase", "P", "V_A", "V_B", "T", "Wcat", "Y_C", "best_so_far"])
        for i, (x, y, b) in enumerate(zip(X, Y, best)):
            w.writerow([i + 1, "random", *np.round(unscale(x), 4), f"{y:.6f}", f"{b:.6f}"])


def load_bo_best(path="results/bo_history.csv"):
    """Scaled location of the best BO run, or None if bo_5knobs.py has not been run."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    r = max(rows, key=lambda r: float(r["Y_C"]))
    phys = np.array([float(r[k]) for k in KEYS])
    return (phys - LOW) / (HIGH - LOW), float(r["Y_C"])


def plot_progress(Y, path, n_bo_budget=50):
    """Best yield so far vs number of simulations (log x-axis)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(Y)
    runs = np.arange(1, n + 1)
    best = np.maximum.accumulate(Y)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(runs, Y, s=3, color=COL_SIM_LIGHT, alpha=0.4, lw=0, label="yield of each random simulation")
    ax.step(runs, best, where="post", color=COL_SIM, lw=2, label="best yield so far")
    ax.set_xscale("log")
    ax.set_xlim(1, n * 1.3)
    ax.set_ylim(0, 1.05)
    ax.axvline(n_bo_budget, color="#52514e", lw=0.8, ls=":")
    ax.annotate(f"after {n_bo_budget} simulations: {best[n_bo_budget - 1]:.4f}",
                xy=(n_bo_budget, best[n_bo_budget - 1]), xytext=(n_bo_budget * 1.4, 0.45),
                fontsize=9, color="#52514e", arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    ax.annotate(f"after {n:,} simulations: {best[-1]:.4f}",
                xy=(n, best[-1]), xytext=(n / 40, 0.72),
                fontsize=9, color="#52514e", arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    ax.set_xlabel("Number of simulated experiments (log scale)")
    ax.set_ylabel("Yield of C,  Y_C")
    ax.set_title(f"Brute-force random simulation on the digital twin ({n:,} runs)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def yield_maps(x_ref, marks, grid, path):
    """5 x 5 'corner' figure.  Diagonal (i, i): Y_C vs knob i with the other four knobs held
    at x_ref.  Below the diagonal (row i, column j): heat map of Y_C over knob j (x-axis) and
    knob i (y-axis), the other three held at x_ref.  `marks` = list of (label, x_scaled,
    marker, colour) points drawn on every panel."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g1 = np.linspace(0, 1, 2 * grid)      # finer grid for the 1-D slices
    g2 = np.linspace(0, 1, grid)
    fig, axes = plt.subplots(5, 5, figsize=(11, 10.5))
    im = None
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:                                        # 1-D slice
                ys = []
                for v in g1:
                    x = x_ref.copy(); x[i] = v; ys.append(f(x))
                ax.plot(unscale_axis(i, g1), ys, color=COL_SIM, lw=1.8)
                ax.set_ylim(0, 1.05)
                for label, xm, mk, col in marks:
                    ax.plot(unscale_axis(i, xm[i]), f(xm), mk, color=col, ms=8, mec="white", mew=0.8)
                ax.set_ylabel("Y_C", fontsize=8)
            else:                                             # 2-D heat map
                Z = np.empty((grid, grid))
                for a, va in enumerate(g2):                   # rows: knob i
                    for b, vb in enumerate(g2):               # cols: knob j
                        x = x_ref.copy(); x[i] = va; x[j] = vb; Z[a, b] = f(x)
                im = ax.pcolormesh(unscale_axis(j, g2), unscale_axis(i, g2), Z,
                                   cmap="Blues", vmin=0, vmax=1, shading="nearest")
                for label, xm, mk, col in marks:
                    ax.plot(unscale_axis(j, xm[j]), unscale_axis(i, xm[i]), mk,
                            color=col, ms=9, mec="white", mew=0.8)
                ax.set_ylabel(NAMES[i], fontsize=8)
            ax.set_xlim(LOW[j], HIGH[j])
            ax.tick_params(labelsize=7)
            if i == 4:
                ax.set_xlabel(NAMES[j], fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    # legend and colour bar go in the empty upper-right triangle
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col, ms=9, mec="white", label=label)
               for label, _, mk, col in marks]
    axes[1, 4].legend(handles=handles, loc="center", fontsize=9, frameon=False)
    ref_txt = ", ".join(f"{k}={v:.3g}" for k, v in zip(KEYS, unscale(x_ref)))
    fig.suptitle("Digital twin: yield over every pair of knobs\n"
                 f"(other knobs held at the best simulated point: {ref_txt})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cax = fig.add_axes([0.80, 0.66, 0.015, 0.24])      # added after tight_layout on purpose
    fig.colorbar(im, cax=cax).set_label("Yield of C,  Y_C")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def unscale_axis(i, v):
    """Scaled value(s) of knob i -> physical units for axis labelling."""
    return LOW[i] + np.asarray(v) * (HIGH[i] - LOW[i])


def sensitivity_oat(base=BASE, n=41):
    """One-at-a-time swing: sweep each knob over its full range with the other four at the
    base case; swing = max(Y_C) - min(Y_C).  Big swing = the yield is very sensitive."""
    swing = []
    for i, key in enumerate(KEYS):
        ys = []
        for v in np.linspace(LOW[i], HIGH[i], n):
            kw = dict(base); kw[key] = v
            ys.append(simulate(V_I=V_I, **kw)["Y_C"])
        swing.append(max(ys) - min(ys))
    return np.array(swing)


def sensitivity_global(X, Y, bins=10):
    """Global first-order sensitivity from the random campaign: for knob i, bin its values,
    take the mean yield per bin, and divide the variance of those bin means by the total
    variance of Y.  This is a simple estimate of the Sobol first-order index S_i
    (share of the yield variance explained by knob i on its own)."""
    s = []
    for i in range(5):
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(X[:, i], edges) - 1, 0, bins - 1)
        means = np.array([Y[idx == b].mean() for b in range(bins)])
        counts = np.array([(idx == b).sum() for b in range(bins)])
        s.append(np.average((means - Y.mean()) ** 2, weights=counts) / Y.var())
    return np.array(s)


def plot_sensitivity(swing, s_glob, path):
    """Two bar panels: one-at-a-time swing at the base case, and global variance share."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    y = np.arange(5)
    for ax, vals, title, xlabel in [
        (a1, swing, "One-at-a-time swing at the base case\n(P=10 bar, 20/20/10 mL/min, 600 K, 1 g)",
         "max Y_C - min Y_C when only this knob is swept"),
        (a2, s_glob, "Global first-order sensitivity\n(from the random campaign, all knobs varying)",
         "share of yield variance explained by this knob"),
    ]:
        ax.barh(y, vals, color=COL_SIM, height=0.55)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color="#52514e")
        ax.set_yticks(y); ax.set_yticklabels(NAMES, fontsize=9); ax.invert_yaxis()
        ax.set_xlim(0, max(vals.max() * 1.25, 0.1))
        ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="x", color="#e5e4e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("How much each knob affects the yield (reference only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="number of random simulations")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=30, help="points per axis in the heat maps")
    args = ap.parse_args()

    print(f"Random simulation campaign: {args.n} experiments")
    X, Y = run_campaign(args.n, args.seed)
    i_best = Y.argmax()
    print(f"\nBest simulated Y_C = {Y.max():.4f} (run {i_best + 1} of {args.n}) at:")
    for name, v in zip(NAMES, unscale(X[i_best])):
        print(f"   {name:14s} {v:8.2f}")
    save_history(X, Y, "results/simulation_history.csv")
    plot_progress(Y, "figures/simulation_progress.png")

    marks = [(f"best of {args.n:,} random simulations (Y_C={Y.max():.3f})", X[i_best], "*", COL_SIM)]
    bo = load_bo_best()
    if bo is not None:
        marks.append((f"best found by BO in 50 runs (Y_C={bo[1]:.3f})", bo[0], "o", COL_BO))
    else:
        print("(results/bo_history.csv not found - run bo_5knobs.py first to mark the BO best)")
    print("\nDrawing 5-knob yield maps ...")
    yield_maps(X[i_best], marks, args.grid, "figures/twin_5param_maps.png")

    print("Computing sensitivities ...")
    swing, s_glob = sensitivity_oat(), sensitivity_global(X, Y)
    for name, a, b in zip(NAMES, swing, s_glob):
        print(f"   {name:14s} OAT swing = {a:.3f}   global S1 = {b:.3f}")
    plot_sensitivity(swing, s_glob, "figures/sensitivity_bar.png")
    print("\nSaved results/simulation_history.csv, figures/simulation_progress.png, "
          "figures/twin_5param_maps.png, figures/sensitivity_bar.png")


if __name__ == "__main__":
    main()
