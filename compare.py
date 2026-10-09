"""
Overlay comparison: Bayesian optimisation vs brute-force random simulation.

Reads the two history files written by bo_5knobs.py and simulation_campaign.py and draws
them on the same axes, so one picture shows "simulating 20,000 random experiments gave
this best yield, BO got here in 50".

Outputs
    figures/overlay_bo_vs_simulation.png   best yield so far vs number of reactor evaluations
    figures/evaluations_to_target.png      evaluations each method needed to reach a target yield
    (a summary table is also printed)

Run:   python bo_5knobs.py  ->  python simulation_campaign.py  ->  python compare.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bo_5knobs import NAMES, COL_BO, COL_INIT
from simulation_campaign import COL_SIM, KEYS

TARGETS = [0.90, 0.95, 0.98, 0.99]    # yield levels used for "evaluations to target"


def load_history(path):
    """Read a history CSV into a dict of numpy arrays (one key per column)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run bo_5knobs.py and simulation_campaign.py first")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    for k in ("run", "P", "V_A", "V_B", "T", "Wcat", "Y_C", "best_so_far"):
        out[k] = out[k].astype(float)
    return out


def evaluations_to_target(best, target):
    """First evaluation number at which best-so-far >= target, or None if never reached."""
    hit = np.nonzero(best >= target)[0]
    return int(hit[0]) + 1 if len(hit) else None


def plot_overlay(bo, sim, path):
    """Both best-so-far curves on one log-x axis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_bo, n_sim = len(bo["run"]), len(sim["run"])
    n_init = int((bo["phase"] == "initial").sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvspan(0.8, n_init + 0.5, color=COL_INIT, alpha=0.25, lw=0,
               label=f"BO initial random design ({n_init} runs)")
    ax.step(sim["run"], sim["best_so_far"], where="post", color=COL_SIM, lw=2,
            label=f"brute-force random simulation ({n_sim:,} runs)")
    ax.step(bo["run"], bo["best_so_far"], where="post", color=COL_BO, lw=2.2,
            label=f"Bayesian optimisation ({n_bo} runs)")
    ax.scatter(bo["run"], bo["Y_C"], s=16, color=COL_BO, zorder=3, lw=0)
    ax.axhline(sim["best_so_far"][-1], color=COL_SIM, lw=0.8, ls=":")
    ax.axvline(n_bo, color="#52514e", lw=0.8, ls=":")

    arrow = dict(arrowstyle="-", color="#52514e", lw=0.8, shrinkB=3)
    ax.annotate(f"BO best = {bo['best_so_far'][-1]:.4f} after {n_bo} runs",
                xy=(n_bo, bo["best_so_far"][-1]), xytext=(n_bo * 1.5, 0.72), fontsize=9,
                color="#0b0b0b", arrowprops=arrow)
    ax.annotate(f"random search, same budget = {sim['best_so_far'][n_bo - 1]:.4f} after {n_bo} runs",
                xy=(n_bo, sim["best_so_far"][n_bo - 1]), xytext=(n_bo * 1.5, 0.60), fontsize=9,
                color="#0b0b0b", arrowprops=arrow)
    ax.annotate(f"random best = {sim['best_so_far'][-1]:.4f} after {n_sim:,} runs",
                xy=(n_sim, sim["best_so_far"][-1]), xytext=(n_sim / 25, 0.86), fontsize=9,
                color="#0b0b0b", ha="right", arrowprops=arrow)
    ax.set_xscale("log")
    ax.set_xlim(0.8, n_sim * 1.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of reactor evaluations (log scale)")
    ax.set_ylabel("Best yield of C found so far,  Y_C")
    ax.set_title("Finding the best operating point: Bayesian optimisation vs brute-force simulation")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_targets(bo, sim, path):
    """Grouped bars: evaluations needed by each method to first reach each target yield."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb = [evaluations_to_target(bo["best_so_far"], t) for t in TARGETS]
    ns = [evaluations_to_target(sim["best_so_far"], t) for t in TARGETS]
    x = np.arange(len(TARGETS)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for off, vals, col, label in [(-w / 2, nb, COL_BO, "Bayesian optimisation"),
                                  (+w / 2, ns, COL_SIM, "random simulation")]:
        hv = [v if v else 0.5 for v in vals]
        ax.bar(x + off, hv, width=w - 0.04, color=col, label=label)
        for xi, v in zip(x + off, vals):
            ax.text(xi, (v if v else 0.5) * 1.15, f"{v}" if v else "not\nreached",
                    ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max([v for v in nb + ns if v] + [10]) * 4)
    ax.set_xticks(x); ax.set_xticklabels([f"Y_C >= {t:.2f}" for t in TARGETS])
    ax.set_ylabel("Reactor evaluations needed (log scale)")
    ax.set_title("Evaluations needed to first reach a target yield")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return nb, ns


def main():
    bo = load_history("results/bo_history.csv")
    sim = load_history("results/simulation_history.csv")
    n_bo, n_sim = len(bo["run"]), len(sim["run"])

    plot_overlay(bo, sim, "figures/overlay_bo_vs_simulation.png")
    nb, ns = plot_targets(bo, sim, "figures/evaluations_to_target.png")

    print(f"{'':34s}{'BO':>12s}{'random sim':>14s}")
    print(f"{'evaluations run':34s}{n_bo:>12d}{n_sim:>14,d}")
    print(f"{'best Y_C after ' + str(n_bo) + ' evaluations':34s}"
          f"{bo['best_so_far'][-1]:>12.4f}{sim['best_so_far'][n_bo - 1]:>14.4f}")
    print(f"{'best Y_C after all evaluations':34s}{bo['best_so_far'][-1]:>12.4f}{sim['best_so_far'][-1]:>14.4f}")
    for t, a, b in zip(TARGETS, nb, ns):
        print(f"{'evaluations to reach Y_C >= ' + f'{t:.2f}':34s}{str(a):>12s}{str(b):>14s}")

    ib, is_ = bo["Y_C"].argmax(), sim["Y_C"].argmax()
    print("\nBest operating point found by each method:")
    print(f"{'':16s}{'BO':>12s}{'random sim':>14s}")
    for name, k in zip(NAMES, KEYS):
        print(f"{name:16s}{bo[k][ib]:>12.2f}{sim[k][is_]:>14.2f}")
    print(f"{'Y_C':16s}{bo['Y_C'][ib]:>12.4f}{sim['Y_C'][is_]:>14.4f}")
    print("\nSaved figures/overlay_bo_vs_simulation.png and figures/evaluations_to_target.png")


if __name__ == "__main__":
    main()
