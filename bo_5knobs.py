"""
Bayesian optimisation (BO) of the reactor digital twin over all 5 knobs.

Knobs (each scaled to 0-1 inside the optimiser):
    P (bar), V_A (mL/min), V_B (mL/min), T (K), Wcat (g).    Inert fixed at V_I = 10 mL/min.
Objective: yield of C,  Y_C = F_C,out / F_A,0   (maximise).

Method: Gaussian-process (GP) surrogate with an RBF kernel whose length-scale is fixed by
hand, a zero-mean prior on standardised yields, and the Expected-Improvement (EI)
acquisition function maximised over random candidate points.  Kernel hyper-parameters
are NOT fitted yet - that is the next step.

Outputs
    results/bo_history.csv    one row per reactor evaluation, in the order they were run
    figures/bo_progress.png   yield of every evaluation, best-so-far curve, and the knob
                              settings BO chose at each step

Run:   python bo_5knobs.py
"""
import os
import csv
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reactor_twin import simulate

# ------------------------------------------------------------------ problem definition
NAMES = ["P (bar)", "V_A (mL/min)", "V_B (mL/min)", "T (K)", "Wcat (g)"]
LOW   = np.array([1.0,   1.0,  1.0,  450.0, 0.05])   # lower bound of each knob
HIGH  = np.array([100.0, 50.0, 50.0, 850.0, 5.0])    # upper bound of each knob
V_I   = 10.0                                          # inert flow, fixed (mL/min)

# ------------------------------------------------------------------ BO settings
N_INIT = 10      # random experiments run before BO starts (initial design)
N_ITER = 40      # experiments chosen by BO
N_CAND = 5000    # random candidate points scored by EI in every iteration
LENGTH = 0.3     # RBF length-scale in scaled (0-1) units, same for every knob
JITTER = 1e-6    # added to the diagonal of K so the linear solve is stable
SEED   = 0

COL_BO, COL_INIT = "#2a78d6", "#9ec5f4"   # BO colours, reused by compare.py


def unscale(x):
    """Scaled knobs in [0, 1]^5  ->  physical units in the order P, V_A, V_B, T, Wcat."""
    return LOW + np.asarray(x) * (HIGH - LOW)


def f(x):
    """Black-box objective: run the digital twin once at scaled point x and return Y_C."""
    P, VA, VB, T, W = unscale(x)
    return simulate(V_A=VA, V_B=VB, V_I=V_I, Wcat=W, T=T, P=P)["Y_C"]


def kernel(a, b, length=LENGTH):
    """RBF (squared-exponential) kernel between the rows of a (n x 5) and b (m x 5) -> n x m.
    k(a, b) = exp(-|a - b|^2 / (2 l^2)):  points close together in all 5 knobs are assumed
    to have similar yields.  k(x, x) = 1, so the prior variance is 1."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * length ** 2))


def gp_posterior(X, Ys, cand, length=LENGTH):
    """Zero-mean GP conditioned on the data (X, Ys).  Returns the posterior mean mu and
    standard deviation sigma at every candidate point:
        mu    = k(cand, X) K^-1 Ys
        sigma = sqrt( 1 - k(cand, X) K^-1 k(X, cand) )          with K = k(X, X) + jitter I
    """
    K = kernel(X, X, length) + JITTER * np.eye(len(X))
    k = kernel(X, cand, length)
    mu = k.T @ np.linalg.solve(K, Ys)
    var = 1.0 - np.sum(k * np.linalg.solve(K, k), axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu, sigma


def expected_improvement(mu, sigma, best):
    """EI(x) = E[ max(f(x) - best, 0) ] = (mu - best) Phi(z) + sigma phi(z),
    z = (mu - best) / sigma,  Phi = normal CDF,  phi = normal PDF."""
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


def run_bo(n_init=N_INIT, n_iter=N_ITER, n_cand=N_CAND, length=LENGTH, seed=SEED, verbose=True):
    """The BO loop.  Returns X (n x 5, scaled inputs) and Y (n yields) in the order they
    were evaluated; the first n_init rows are the random initial design."""
    np.random.seed(seed)
    X = np.random.rand(n_init, 5)
    Y = np.array([f(x) for x in X])
    for i in range(n_iter):
        Ys = (Y - Y.mean()) / Y.std()            # standardise: unit-variance prior now makes sense
        cand = np.random.rand(n_cand, 5)         # fresh random candidates (no grid in 5-D)
        mu, sigma = gp_posterior(X, Ys, cand, length)
        ei = expected_improvement(mu, sigma, Ys.max())
        x_next = cand[np.argmax(ei)]             # experiment with the highest EI
        y_next = f(x_next)
        X, Y = np.vstack([X, x_next]), np.append(Y, y_next)
        if verbose:
            print(f"run {n_init + i + 1:2d}: Y_C = {y_next:.4f}, best = {Y.max():.4f}")
    return X, Y


def save_history(X, Y, n_init, path):
    """CSV with one row per evaluation: run number, phase, the 5 knobs in physical
    units, the yield, and the best yield seen up to that run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best = np.maximum.accumulate(Y)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "phase", "P", "V_A", "V_B", "T", "Wcat", "Y_C", "best_so_far"])
        for i, (x, y, b) in enumerate(zip(X, Y, best)):
            phase = "initial" if i < n_init else "bo"
            w.writerow([i + 1, phase, *np.round(unscale(x), 4), f"{y:.6f}", f"{b:.6f}"])


def plot_progress(X, Y, n_init, path):
    """Top: yield of every evaluation and the best-so-far curve.
    Bottom: the knob settings chosen at every evaluation (0 = low bound, 1 = high bound)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(Y)
    runs = np.arange(1, n + 1)
    best = np.maximum.accumulate(Y)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.3]})

    ax.axvspan(0.5, n_init + 0.5, color=COL_INIT, alpha=0.25, lw=0,
               label=f"initial random design ({n_init} runs)")
    ax.step(runs, best, where="post", color=COL_BO, lw=2, label="best yield so far")
    ax.scatter(runs[:n_init], Y[:n_init], s=28, color=COL_INIT, edgecolor=COL_BO, zorder=3)
    ax.scatter(runs[n_init:], Y[n_init:], s=28, color=COL_BO, zorder=3,
               label="yield of each BO-chosen run")
    ax.annotate(f"best Y_C = {Y.max():.4f}\n(run {Y.argmax() + 1})",
                xy=(Y.argmax() + 1, Y.max()), xytext=(0.62, 0.35), textcoords="axes fraction",
                fontsize=9, color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    ax.set_ylabel("Yield of C,  Y_C")
    ax.set_ylim(0, 1.05)
    ax.set_title("Bayesian optimisation on the digital twin: progress over 50 reactor evaluations")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    im = ax2.imshow(X.T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                    extent=(0.5, n + 0.5, 4.5, -0.5))
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(NAMES, fontsize=8)
    ax2.set_xlabel("Reactor evaluation number")
    ax2.set_title("Knob setting chosen at each evaluation (0 = lower bound, 1 = upper bound)",
                  fontsize=9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cb.set_label("scaled setting", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    X, Y = run_bo()
    best_x = unscale(X[Y.argmax()])
    print(f"\nBO best Y_C = {Y.max():.4f} after {len(Y)} runs, at:")
    for name, v in zip(NAMES, best_x):
        print(f"   {name:14s} {v:8.2f}")
    save_history(X, Y, N_INIT, "results/bo_history.csv")
    plot_progress(X, Y, N_INIT, "figures/bo_progress.png")
    print("\nSaved results/bo_history.csv and figures/bo_progress.png")
    print("Run simulation_campaign.py and then compare.py for the brute-force comparison.")


if __name__ == "__main__":
    main()
