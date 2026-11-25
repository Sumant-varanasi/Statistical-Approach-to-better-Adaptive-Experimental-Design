"""Preliminary v7 comparison figure from the laptop cache (results/smart_bench) and the HPC run (results/hpc_early).

Campaign files are read only if they were made by the PRESENT definition of their method (smart.benchmark.is_stale):
the results/hpc_early smart_v7* files were made when v7 meant the hybrid committee without network revision, and its
nn_only files under an earlier recommendation rule, so they are skipped and counted in the footer.  A method with no
valid file is left out of the figure.  results/smart_bench is searched first."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smart.advisor import true_optimum
from smart import benchmark as B

ROOTS = ["results/smart_bench", "results/hpc_early/bench"]
SKIPPED = set()                 # stale files met while loading
SEEDS, BUDGET = 10, 25
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8986", "#e6e5e1", "#fcfcfb"
COL = {"nn_only": "#eb6834", "plain_bo": "#e87ba4", "chemist": "#eda100", "smart": "#2a78d6",
       "smart_v6": "#1baf7a", "smart_v7": "#008300"}
NAME = {"nn_only": "NN alone (best run measured)", "plain_bo": "Plain BO (GP, profit only)", "chemist": "Rules of thumb (script)",
        "smart": "v5 advisor (physics + GP)", "smart_v6": "v6 advisor (physics + GP + NNs)",
        "smart_v7": "v7 advisor (hybrid + revision)"}
ORDER = ["nn_only", "plain_bo", "chemist", "smart", "smart_v6", "smart_v7"]
SCEN = [("simple", "", "Simple reaction"), ("complex", "", "Complex reaction, model correct"),
        ("complex", "_missing_E", "Complex, C+B→E missing from the model"),
        ("complex_deact", "", "Complex, catalyst ageing")]
MODEL_FREE = {"nn_only", "plain_bo", "chemist"}


def load(st, m, k):
    """The campaign file of method m, setting st, seed k: the first one found that the present definition of m made."""
    for r in ROOTS:
        p = os.path.join(r, st, f"{m}_s{k}_b{BUDGET}.npz")
        if os.path.exists(p):
            if B.is_stale(p, m):
                SKIPPED.add(p)
                continue
            return np.load(p, allow_pickle=True)
    return None


def method_name(m, suffix):
    return m if (m in MODEL_FREE or not suffix) else m + suffix


def campaigns(st, m):
    f = true_optimum(B.SETTINGS[st]["system"])["f_opt"]
    out = [load(st, m, k) for k in range(SEEDS)]
    out = [d for d in out if d is not None]
    return f, out


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(19, 10.5), facecolor=SURF)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)

# ---------------------------------------------------------------- row 1: % of best profit vs runs
for j, (st, suf, title) in enumerate(SCEN):
    ax = fig.add_subplot(gs[0, j])
    style(ax)
    ax.axvspan(0.5, 5.5, color="#efeeea", lw=0)
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    ends = []
    for m in ORDER:
        mm = method_name(m, suf)
        f, C = campaigns(st, mm)
        if not C:
            continue
        R = np.array([100 * d["rec"] / f for d in C])
        med = np.maximum(np.median(R, 0), -60)
        n = np.arange(1, R.shape[1] + 1)
        ax.plot(n, med, color=COL[m], lw=2.2 if m.startswith("smart") else 1.6, label=NAME[m], solid_capstyle="round")
        ends.append((med[-1], m))
    ax.set_ylim(-63, 112)
    ax.set_xlim(0.5, BUDGET + 0.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("runs on the reactor", fontsize=8, color=INK2)
    if j == 0:
        ax.set_ylabel("profit of the recommended settings\n(% of the best possible, median of 10)", fontsize=8, color=INK2)
        ax.text(3, -58, "5 shared\nscreening runs", fontsize=7, color=INK2, ha="center")
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.975))

# ---------------------------------------------------------------- row 2, panel 1: runs to 90 % (table)
ax = fig.add_subplot(gs[1, 0:2])
ax.set_facecolor(SURF)
ax.axis("off")
cols = ["Simple reaction", "Complex,\nmodel correct", "Complex,\nC+B→E missing", "Complex,\ncatalyst ageing"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#1c5cab", "#86b6ef", "#e8f1fc"])
ax.text(0.0, 1.03, "Runs needed to reach 90 % of the best profit", fontsize=10, color=INK, transform=ax.transAxes)
ax.text(0.0, 0.975, "median of 10 campaigns, fewer is better; small print = campaigns that got there within 25 runs",
        fontsize=7.5, color=INK2, transform=ax.transAxes)
x0, cw, y0, rh = 0.30, 0.175, 0.83, 0.125
for j, c in enumerate(cols):
    ax.text(x0 + (j + 0.5) * cw, y0 + 0.07, c, ha="center", va="center", fontsize=8, color=INK2, transform=ax.transAxes)
for i, m in enumerate(ORDER):
    yy = y0 - (i + 1) * rh
    ax.add_patch(plt.Rectangle((0.0, yy + rh * 0.3), 0.012, rh * 0.4, color=COL[m], transform=ax.transAxes, lw=0))
    ax.text(0.02, yy + rh / 2, NAME[m], fontsize=8.5, color=INK, va="center", transform=ax.transAxes)
    for j, (st, suf, _) in enumerate(SCEN):
        f, C = campaigns(st, method_name(m, suf))
        if not C:
            ax.text(x0 + (j + 0.5) * cw, yy + rh / 2, "no data", ha="center", va="center", fontsize=8, color=MUTED,
                    transform=ax.transAxes)
            continue
        t = []
        for d in C:
            r = 100 * d["rec"] / f
            hit = np.nonzero(r >= 90)[0]
            t.append(hit[0] + 1 if len(hit) else np.inf)
        t = np.array(t, float)
        med, reached = np.median(t), int(np.isfinite(t).sum())
        if np.isfinite(med):
            fc, big, tc = cmap(float(min(1.0, (med - 8.0) / 14.0))), f"{med:g}", ("#ffffff" if med < 13 else INK)
        else:
            fc, big, tc = "#f1f0ec", "never", INK2
        ax.add_patch(plt.Rectangle((x0 + j * cw + 0.003, yy + 0.008), cw - 0.006, rh - 0.016, color=fc,
                                   transform=ax.transAxes, lw=0))
        ax.text(x0 + (j + 0.5) * cw, yy + rh * 0.6, big, ha="center", va="center", fontsize=10,
                color=tc, transform=ax.transAxes, fontweight="bold" if np.isfinite(med) else "normal")
        ax.text(x0 + (j + 0.5) * cw, yy + rh * 0.25, f"{reached}/{len(t)}", ha="center", va="center", fontsize=7,
                color=tc, transform=ax.transAxes)

# ---------------------------------------------------------------- row 2, panel 2: contribution of each part
ax = fig.add_subplot(gs[1, 2])
style(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, lw=0.8)
PARTS = [("smart_missing_E", "v5: physics + GP", "#2a78d6"),
         ("smart_v6_nonn_missing_E", "v6 without its NNs (physics + GP, weighted)", "#86b6ef"),
         ("smart_v6_nogp_missing_E", "v6 without its GP (physics + NNs)", "#8fd5bb"),
         ("smart_v6_missing_E", "v6: physics + GP + NNs", "#1baf7a"),
         ("smart_v7_missing_E", "v7: v6 + neural network alone", "#008300")]
for i, (m, lab, c) in enumerate(PARTS):
    _, C = campaigns("complex", m)
    y = len(PARTS) - 1 - i
    if not C:
        ax.text(-115, y + 0.36, lab, fontsize=7.8, color=MUTED, va="center")
        ax.text(5, y, "no data of the present definition yet", fontsize=7.5, color=MUTED, va="center")
        continue
    E = np.array([d["cum_true"][-1] for d in C])
    q1, med, q3 = np.percentile(E, [25, 50, 75])
    ax.barh(y, med, height=0.42, color=c)
    ax.plot([q1, q3], [y, y], color=INK, lw=1.1)
    ax.text(-115, y + 0.36, lab, fontsize=7.8, color=INK, va="center")
    ax.text(max(med, q3) + 5, y, f"{med:.0f}", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks([])
ax.set_xlim(-118, 185)
ax.set_ylim(-0.5, len(PARTS) - 0.2)
ax.set_title("What each part adds, C+B→E missing", fontsize=10, color=INK, loc="left")
ax.set_xlabel("profit earned over the 25-run campaign\n(median of 10; black line = middle half of the campaigns)", fontsize=8, color=INK2)

# ---------------------------------------------------------------- row 2, panel 3: held-out prediction error
ax = fig.add_subplot(gs[1, 3])
style(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, lw=0.8)
MEM = [("none", "physics alone", MUTED), ("gp", "physics + GP", "#2a78d6"),
       ("nn", "physics + NNs", "#1baf7a"), ("direct", "neural network alone", "#eb6834")]
y = 0
ticks = []
for m, title in [("smart_v7_missing_E", "C+B→E missing from the model"), ("smart_v7", "Model correct")]:
    _, C = campaigns("complex", m)
    if not C:
        ax.text(2.6, y + 1.5, f"{title}: no v7 campaigns of the present definition yet\n(run benchmark.run_all(methods=benchmark.V7_STUDY))",
                fontsize=7.5, color=MUTED)
        y += 4.2
        continue
    E = np.array([d["committee_heldout_rmse"] for d in C])          # campaigns x updates x 4
    cols4 = list(C[0]["weight_columns"])
    for key, lab, c in MEM[::-1]:
        v = np.nanmean(E[:, -5:, cols4.index(key)], 1)               # last 5 updates (20-24 runs)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.barh(y, med, height=0.62, color=c)
        ax.plot([q1, q3], [y, y], color=INK, lw=1.1)
        ax.text(max(med, q3) + 0.4, y, f"{med:.1f}  {lab}", va="center", fontsize=7.5, color=INK)
        y += 1
    ax.text(0, y - 0.15, title, fontsize=8.5, color=INK, fontweight="bold")
    y += 1.2
ax.axvline(2.1, color=INK2, lw=1, ls="--")
ax.text(2.5, y - 0.4, "← measurement noise (GC error → profit)", fontsize=7, color=INK2, va="center")
ax.set_yticks([])
ax.set_ylim(-0.8, y + 0.2)
ax.set_xlim(0, 33)
ax.set_title("Error predicting runs it has not seen", fontsize=10, color=INK, loc="left")
ax.set_xlabel("typical error of the predicted profit (money/h)\nheld-out runs after 20–24 runs, median of 10", fontsize=8, color=INK2)

fig.text(0.01, 0.005, "PRELIMINARY. Only campaign files made by the present definition of each method are shown "
         f"({len(SKIPPED)} older files skipped, e.g. the HPC run of 23 Nov for v7 and the neural network alone); v5/v6 from the "
         "committed benchmark. Simulated reactor, placeholder kinetics and prices. Curves clipped at -60 %. "
         "NN = neural network, GP = Gaussian process.", fontsize=7.5, color=INK2)
out = "figures/smart_7_prelim_campaigns.png"
fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
print("->", out)
if SKIPPED:
    print(f"skipped {len(SKIPPED)} file(s) made by an earlier definition of their method, e.g. {sorted(SKIPPED)[0]}")
