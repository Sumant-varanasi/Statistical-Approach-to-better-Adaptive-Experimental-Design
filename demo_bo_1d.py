"""
Teaching figure: what Bayesian optimisation is actually doing, one step at a time.

One knob (temperature), everything else at the base case of sweeps.py, objective = yield.
Each row of the figure is one BO iteration:
    left   the true yield curve (from the twin, normally unknown), the experiments done so far,
           the GP's best guess (mean) and its 95 % uncertainty band
    right  the acquisition functions EI and UCB across temperature (scaled to 0-1 for display);
           the arrow marks where EI sends the next experiment
Output:  figures/bo_1d_steps.png
Run:     python demo_bo_1d.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reactor_twin import simulate
from bo.gp import GP
from bo.acquisition import expected_improvement, upper_confidence_bound, ucb_beta

T_LOW, T_HIGH = 450.0, 850.0
BLUE, ORANGE, VIOLET, INK2, GRID = "#2a78d6", "#eb6834", "#6250d6", "#52514e", "#e1e0d9"


def f(x):
    T = T_LOW + x * (T_HIGH - T_LOW)
    return simulate(V_A=20, V_B=20, V_I=10, Wcat=1.0, T=T, P=10)["Y_C"]


def main(n_steps=4):
    grid = np.linspace(0, 1, 400)
    truth = np.array([f(x) for x in grid])
    Tg = T_LOW + grid * (T_HIGH - T_LOW)
    X = np.array([0.05, 0.35, 0.6, 0.95])   # 4 starts: with only 3 the fitted GP can over-fit
    Y = np.array([f(x) for x in X])
    fig, axes = plt.subplots(n_steps, 2, figsize=(11, 2.6 * n_steps), sharex=True,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for i in range(n_steps):
        gp = GP(seed=0).fit(X[:, None], Y)
        mu, sd = gp.predict(grid[:, None])
        ei = expected_improvement(mu, sd, Y.max())
        ucb = upper_confidence_bound(mu, sd, ucb_beta(len(X) + 1, 1))
        x_next = grid[np.argmax(ei)]

        ax = axes[i, 0]
        ax.plot(Tg, truth, color="#898781", lw=1.2, ls="--", label="true yield (twin)")
        ax.fill_between(Tg, mu - 1.96 * sd, mu + 1.96 * sd, color=BLUE, alpha=0.15, lw=0,
                        label="GP 95 % band")
        ax.plot(Tg, mu, color=BLUE, lw=2, label="GP mean")
        ax.scatter(T_LOW + X * (T_HIGH - T_LOW), Y, color=ORANGE, s=36, zorder=3,
                   edgecolor="white", label="experiments so far")
        ax.axvline(T_LOW + x_next * (T_HIGH - T_LOW), color=INK2, lw=0.8, ls=":")
        ax.set_ylabel("Y_C", fontsize=8)
        ax.set_title(f"Step {i + 1}: {len(X)} experiments, best Y_C = {Y.max():.3f}", fontsize=9)

        ax2 = axes[i, 1]
        ax2.plot(Tg, ei / max(ei.max(), 1e-12), color=BLUE, lw=2, label="EI (scaled)")
        ax2.plot(Tg, (ucb - ucb.min()) / max(np.ptp(ucb), 1e-12), color=VIOLET, lw=1.5, ls="--",
                 label="UCB (scaled)")
        ax2.annotate("next", xy=(T_LOW + x_next * (T_HIGH - T_LOW), 1), xytext=(0, 12),
                     textcoords="offset points", ha="center", fontsize=8, color=INK2,
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
        ax2.set_ylim(0, 1.25)
        ax2.set_title("Acquisition: where to experiment next", fontsize=9)
        for a in (ax, ax2):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.grid(axis="y", color=GRID, lw=0.6)
            a.tick_params(labelsize=7)
        X, Y = np.append(X, x_next), np.append(Y, f(x_next))
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1, 0].set_xlabel("Reactor temperature T (K)", fontsize=8)
    axes[-1, 1].set_xlabel("Reactor temperature T (K)", fontsize=8)
    fig.suptitle("Bayesian optimisation step by step (one knob: temperature)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("figures/bo_1d_steps.png", dpi=150)
    plt.close(fig)
    print(f"Saved figures/bo_1d_steps.png  (true best Y_C on the grid = {truth.max():.4f}, "
          f"BO best after {len(Y)} experiments = {Y.max():.4f})")


if __name__ == "__main__":
    main()
