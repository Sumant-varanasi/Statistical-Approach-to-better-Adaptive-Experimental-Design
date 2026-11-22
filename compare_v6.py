"""
v6 vs v5: paired comparison on the cached benchmark campaigns (results/smart_bench).

    python compare_v6.py            -> results/v6_summary.md, figures/smart_6_v6_vs_v5.png

Every v6 campaign has a v5 twin with the same seed: the same virtual reactor, the same GC noise
and the same five screening runs, so the two can be compared campaign by campaign.  Reported:
the true profit of the recommended settings (% of the best possible on a fresh catalyst) after
10 / 15 runs and at the end, the runs needed to reach 90 %, the profit earned over the campaign,
paired wins and a Wilcoxon signed-rank test, and how much weight v6 gave each committee member.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from smart import benchmark as B
from smart.advisor import true_optimum

CASES = [  # (label, setting, budget, v5 method, v6 method)
    ("Simple reaction, 25 runs", "simple", 25, "smart", "smart_v6"),
    ("Complex reaction, 25 runs", "complex", 25, "smart", "smart_v6"),
    ("Complex, ageing catalyst, 25 runs", "complex_deact", 25, "smart", "smart_v6"),
    ("Complex, C+B->E missing from the model, 25 runs", "complex", 25, "smart_missing_E", "smart_v6_missing_E"),
    ("Complex, ageing, C+B->E missing, 25 runs", "complex_deact", 25, "smart_missing_E", "smart_v6_missing_E"),
    ("Complex reaction, 40 runs", "complex", 40, "smart", "smart_v6"),
    ("Complex, C+B->E missing, 40 runs", "complex", 40, "smart_missing_E", "smart_v6_missing_E"),
]
SEEDS = 10


def runs_to(rel, level=90.0):
    hit = np.nonzero(rel >= level)[0]
    return float(hit[0] + 1) if len(hit) else np.inf


def pair(setting, budget, m5, m6):
    f = true_optimum(B.SETTINGS[setting]["system"])["f_opt"]
    out = []
    for k in range(SEEDS):
        p5, p6 = B.path_for(setting, m5, k, budget), B.path_for(setting, m6, k, budget)
        if os.path.exists(p5) and os.path.exists(p6):
            a, b = np.load(p5, allow_pickle=True), np.load(p6, allow_pickle=True)
            out.append(dict(seed=k, r5=100 * a["rec"] / f, r6=100 * b["rec"] / f, c5=a["cum_true"], c6=b["cum_true"],
                            w=b["committee_weights"] if "committee_weights" in b.files else None))
    return out


def p_value(x, y):
    d = np.asarray(x) - np.asarray(y)
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return 1.0


def main():
    lines = ["# v6 (committee correction) vs v5 (GP correction)", "",
             "Paired campaigns: same seed = same virtual reactor, GC noise and screening runs. Profit = true profit of the "
             "recommended settings, % of the best possible on a fresh catalyst (median over campaigns). Earned = profit "
             "produced over the whole campaign (money units, median). Placeholder kinetics and prices.", "",
             "| Scenario | n | after 10 runs v5 / v6 | after 15 | at the end | runs to 90 % v5 / v6 | earned v5 / v6 | v6 earned more | p (earned) | v6 weights at the end (none / GP / NN) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for ax, (label, st, budget, m5, m6) in zip(axes, CASES):
        P = pair(st, budget, m5, m6)
        if not P:
            lines.append(f"| {label} | 0 | not run | | | | | | | |")
            ax.set_visible(False)
            continue
        R5, R6 = np.array([p["r5"] for p in P]), np.array([p["r6"] for p in P])
        C5, C6 = np.array([p["c5"][-1] for p in P]), np.array([p["c6"][-1] for p in P])
        t5, t6 = np.array([runs_to(r) for r in R5]), np.array([runs_to(r) for r in R6])
        fmt_r = lambda t: (f"{np.median(t):g}" if np.isfinite(np.median(t)) else f">{budget}")
        W = np.array([p["w"][-1] for p in P if p["w"] is not None and len(p["w"])])
        wtxt = " / ".join(f"{100 * v:.0f} %" for v in W.mean(0)) if len(W) else "-"
        lines.append(
            f"| {label} | {len(P)} | {np.median(R5[:, 9]):.0f} / {np.median(R6[:, 9]):.0f} % | "
            f"{np.median(R5[:, 14]):.0f} / {np.median(R6[:, 14]):.0f} % | {np.median(R5[:, -1]):.0f} / {np.median(R6[:, -1]):.0f} % | "
            f"{fmt_r(t5)} / {fmt_r(t6)} | {np.median(C5):.0f} / {np.median(C6):.0f} | {int(np.sum(C6 > C5))}/{len(P)} | "
            f"{p_value(C6, C5):.2g} | {wtxt} |")
        n = np.arange(1, R5.shape[1] + 1)
        for R, col, name in ((R5, "#eb6834", "v5 (GP)"), (R6, "#8B2327", "v6 (committee)")):
            med = np.median(R, 0)
            q1, q3 = np.percentile(R, [25, 75], 0)
            ax.fill_between(n, np.maximum(q1, -60), q3, color=col, alpha=0.12, lw=0)
            ax.plot(n, np.maximum(med, -60), color=col, lw=2, label=name)
        ax.axhline(100, color="#999", lw=0.8, ls=":")
        ax.axvspan(0.5, 5.5, color="#eee", lw=0)
        ax.set_ylim(-62, 110)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("runs", fontsize=8)
        ax.tick_params(labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    # committee weights over a campaign (complex, wrong network, longest budget available)
    ax = axes[-1]
    ax.set_visible(True)
    P = pair("complex", 40, "smart_missing_E", "smart_v6_missing_E") or pair("complex", 25, "smart_missing_E", "smart_v6_missing_E")
    Ws = [p["w"] for p in P if p["w"] is not None and len(p["w"])]
    if Ws:
        L = min(len(w) for w in Ws)
        Wm = np.mean([w[:L] for w in Ws], 0)
        x = np.arange(6, 6 + L)
        ax.stackplot(x, Wm.T, labels=["none", "Gaussian process", "neural networks"],
                     colors=["#bbbbbb", "#eb6834", "#8B2327"], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title("v6 committee weights, C+B->E missing (mean)", fontsize=9)
        ax.set_xlabel("run the advice is for", fontsize=8)
        ax.legend(fontsize=7, loc="lower left", frameon=False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("% of the best possible profit", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/smart_6_v6_vs_v5.png", dpi=140)
    os.makedirs("results", exist_ok=True)
    with open("results/v6_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> results/v6_summary.md, figures/smart_6_v6_vs_v5.png")


if __name__ == "__main__":
    main()
