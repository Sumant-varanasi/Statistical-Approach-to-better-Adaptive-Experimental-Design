"""
Statistics and figures for the multi-seed benchmark (run_benchmark.py must have run first).

Main metric: NORMALISED REGRET of the recommended operating point
    regret_t = (f_opt - f(recommended point after t experiments)) / (f_opt - f_median)
    0  = found the true optimum;   1 = no better than a typical random operating point.
Normalising lets the yield and margin problems share one scale.

Outputs
    figures/bench_convergence.png       median regret vs experiments (main methods), one panel per setting
    figures/bench_acquisition.png       same, GP with EI / UCB / PI / Thompson / batch-of-4
    figures/bench_surrogate.png         same, GP vs NN ensemble, and GP with/without output transform
    figures/bench_final_regret.png      every run's final regret, every strategy, every setting
    figures/bench_evals_to_target.png   experiments needed to get within 5 % and 1 % of the optimum
    results/bench_summary.csv           one row per setting x strategy
    results/bench_tests.csv             significance tests (with Holm correction) vs GP + EI
    results/bench_summary.json          everything above, for the results web page

Run:   python analyse_benchmark.py
"""
import os
import glob
import json
import csv
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bo.loop import STRATEGIES

BENCH = "results/bench"
SETTING_LABEL = {"yield": "Yield of C (easy, no noise)",
                 "margin": "Process margin (hard, no noise)",
                 "margin_noisy": "Process margin with measurement noise",
                 "margin_spec": "Process margin, selectivity must be >= 0.97"}
MAIN_SETTINGS = ["yield", "margin", "margin_noisy"]
TARGETS = [0.05, 0.01]          # normalised regret thresholds for "evaluations to target"
REF = "gp_ei"                   # the method every test compares against
PAIRED = {"bo_v1": False}       # bo_v1 uses its own random start, so it is not paired

# colour follows the method, in every figure (validated categorical palette, fixed order)
COLOR = {"gp_ei": "#2a78d6", "random": "#eb6834", "bo_v1": "#1baf7a", "nn_ei": "#eda100",
         "simplex": "#e87ba4", "lhs": "#008300", "gp_ucb": "#6250d6", "gp_ts": "#e34948",
         "gp_pi": "#008300", "gp_ei_q4": "#e87ba4", "gp_ei_raw": "#1baf7a", "nn_ts": "#eda100",
         "cgp_ei": "#6250d6", "gp_ei_logw": "#6250d6", "gp_ei_logall": "#e34948"}
DASH = {"nn_ts": (4, 2), "gp_ei_raw": (4, 2)}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FLOOR = 1e-4                    # regret below this is drawn at the floor (log axis)

GROUPS = {
    "bench_convergence.png": ["gp_ei", "nn_ei", "bo_v1", "simplex", "lhs", "random"],
    "bench_acquisition.png": ["gp_ei", "gp_ucb", "gp_pi", "gp_ts", "gp_ei_q4"],
    "bench_surrogate.png": ["gp_ei", "gp_ei_raw", "nn_ei", "nn_ts"],
    "bench_constrained.png": ["cgp_ei", "gp_ei", "bo_v1", "lhs", "random"],
    "bench_log_inputs.png": ["gp_ei_logw", "gp_ei_logall", "gp_ei", "bo_v1"],
}
GROUP_SETTINGS = {"bench_constrained.png": ["margin_spec"]}


def load():
    data = {}
    for path in glob.glob(os.path.join(BENCH, "*", "*.npz")):
        setting = os.path.basename(os.path.dirname(path))
        z = np.load(path)
        strat = str(z["strategy"])
        gap = float(z["f_opt"] - z["f_ref"])
        reg = np.maximum((float(z["f_opt"]) - z["rec"]) / gap, 0.0)
        d = data.setdefault(setting, {}).setdefault(strat, {})
        d[int(z["seed"])] = dict(regret=reg, time=float(z["time_total"]),
                                 X=z["X"], f_opt=float(z["f_opt"]))
    return data


def matrix(runs):
    seeds = sorted(runs)
    return seeds, np.array([runs[s]["regret"] for s in seeds])


def evals_to(reg_row, thr):
    hit = np.nonzero(reg_row <= thr)[0]
    return int(hit[0]) + 1 if len(hit) else None


def a12(x, y):
    """Vargha-Delaney A: probability a random run of x has LOWER regret than one of y."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum())
                 / (len(x) * len(y)))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summarise(data):
    rows, tests = [], []
    for setting, strats in data.items():
        for s, runs in strats.items():
            seeds, M = matrix(runs)
            final = M[:, -1]
            row = dict(setting=setting, strategy=s, label=STRATEGIES[s]["label"], n_seeds=len(seeds),
                       budget=M.shape[1], median_final_regret=float(np.median(final)),
                       q25=float(np.percentile(final, 25)), q75=float(np.percentile(final, 75)),
                       mean_time_s=float(np.mean([runs[k]["time"] for k in seeds])))
            for thr in TARGETS:
                e = [evals_to(r, thr) for r in M]
                hit = [v for v in e if v is not None]
                row[f"success_{thr}"] = len(hit) / len(e)
                # median evals, counting failures as "more than the budget" (inf)
                ee = np.array([v if v is not None else np.inf for v in e], float)
                med = np.median(ee)
                row[f"median_evals_{thr}"] = None if np.isinf(med) else float(med)
            rows.append(row)

        # tests: every strategy vs the reference (lower final regret = better)
        if REF not in strats:
            continue
        ref_seeds, ref_M = matrix(strats[REF])
        ref_final = dict(zip(ref_seeds, ref_M[:, -1]))
        block = []
        for s, runs in strats.items():
            if s == REF:
                continue
            seeds, M = matrix(runs)
            other = dict(zip(seeds, M[:, -1]))
            common = sorted(set(ref_final) & set(other))
            x = np.array([ref_final[k] for k in common])
            y = np.array([other[k] for k in common])
            paired = PAIRED.get(s, STRATEGIES[s]["kind"] == "bo")
            if paired and np.any(x != y):
                p = wilcoxon(x, y, alternative="two-sided", zero_method="zsplit").pvalue
                test = "Wilcoxon signed-rank (paired by seed)"
            else:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
                test = "Mann-Whitney U"
            block.append(dict(setting=setting, strategy=s, label=STRATEGIES[s]["label"], test=test,
                              n=len(common), median_ref=float(np.median(x)), median_other=float(np.median(y)),
                              A12_ref_better=a12(x, y), p=float(p)))
        for b, padj in zip(block, holm([b["p"] for b in block])):
            b["p_holm"] = float(padj)
            b["verdict"] = ("GP+EI better" if padj < 0.05 and b["A12_ref_better"] > 0.5 else
                            f"{b['label']} better" if padj < 0.05 else "no significant difference")
        tests += block
    return rows, tests


# ---------------------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def convergence_figure(data, strategies, path, title, settings=None):
    settings = [s for s in (settings or MAIN_SETTINGS) if s in data]
    if not settings:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(max(4.6 * len(settings), 10), 4.4 if len(settings) > 1 else 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, setting in zip(axes, settings):
        ends = []
        for s in strategies:
            if s not in data[setting]:
                continue
            _, M = matrix(data[setting][s])
            M = np.maximum(M, FLOOR)
            t = np.arange(1, M.shape[1] + 1)
            med = np.median(M, 0)
            lo, hi = np.percentile(M, 25, 0), np.percentile(M, 75, 0)
            ax.fill_between(t, lo, hi, color=COLOR[s], alpha=0.12, lw=0)
            ax.plot(t, med, color=COLOR[s], lw=2, dashes=DASH.get(s, ()), label=STRATEGIES[s]["label"])
            ends.append([med[-1], s])
        # direct end labels, nudged apart so they do not collide
        ends.sort(reverse=True)                     # place from the top down, so labels never leave the axis
        ly = [min(np.log10(v), 0.05) for v, _ in ends]
        for i in range(1, len(ly)):
            ly[i] = min(ly[i], ly[i - 1] - 0.21)
        for (v, s), y in zip(ends, ly):
            ax.annotate(STRATEGIES[s]["label"].split(" (")[0], xy=(t[-1], v), xytext=(t[-1] + 1.5, 10 ** y),
                        fontsize=7, color=INK2, va="center",
                        arrowprops=dict(arrowstyle="-", color=COLOR[s], lw=0.8))
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.8, 1.5)
        ax.set_xlim(1, t[-1] + 16)
        ax.axvline(10.5, color=MUTED, lw=0.6, ls=":")
        ax.text(11.2, FLOOR * 1.25, "BO starts after 10 starting experiments", fontsize=6.5, color=MUTED, va="bottom")
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_title(SETTING_LABEL[setting], fontsize=9, color=INK)
        ax.set_xlabel("Experiments run", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("Normalised regret (log scale, lower is better)\n0 = true optimum, 1 = typical random point",
                       fontsize=8, color=INK2)
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6 if len(settings) > 1 else 3), fontsize=8, frameon=False)
    fig.suptitle(title + "  (line = median of 20 seeds, band = middle 50 %)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06 if len(settings) > 1 else 0.1, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def final_regret_figure(data, path):
    settings = [s for s in MAIN_SETTINGS if s in data]
    order = [s for s in STRATEGIES if any(s in data[st] for st in settings)]
    fig, axes = plt.subplots(1, len(settings), figsize=(4.4 * len(settings), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, setting in zip(axes, settings):
        for i, s in enumerate(order):
            if s not in data[setting]:
                continue
            _, M = matrix(data[setting][s])
            v = np.maximum(M[:, -1], FLOOR)
            col = COLOR["gp_ei"] if s == REF else "#898781"
            ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=12, color=col, alpha=0.7, lw=0)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.set_xscale("log")
        ax.set_xlim(FLOOR * 0.7, 3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([STRATEGIES[s]["label"] for s in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(SETTING_LABEL[setting], fontsize=9, color=INK)
        ax.set_xlabel("Final normalised regret after 50 experiments (log)", fontsize=8, color=INK2)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
    fig.suptitle("Where every run ended up (dot = one seed, bar = median). Further left is better.",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def targets_figure(rows, path, strategies):
    settings = [s for s in MAIN_SETTINGS if any(r["setting"] == s for r in rows)]
    fig, axes = plt.subplots(len(TARGETS), len(settings), figsize=(4.4 * len(settings), 3.4 * len(TARGETS)),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for j, setting in enumerate(settings):
        for i, thr in enumerate(TARGETS):
            ax = axes[i, j]
            rs = {r["strategy"]: r for r in rows if r["setting"] == setting}
            names = [s for s in strategies if s in rs]
            for k, s in enumerate(names):
                r = rs[s]
                med = r[f"median_evals_{thr}"]
                succ = r[f"success_{thr}"]
                ax.barh(k, med if med else 0, color=COLOR[s], height=0.6)
                txt = (f"{med:.0f} runs" if med else "not reached by most seeds") + f"   ({succ:.0%} of seeds hit it)"
                ax.text((med or 0) + 0.8, k, txt, va="center", fontsize=7, color=INK2)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([STRATEGIES[s]["label"] for s in names], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 85)
            ax.set_title(f"{SETTING_LABEL[setting]}\nwithin {thr:.0%} of the optimum", fontsize=8.5, color=INK)
            style(ax)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.grid(axis="y", visible=False)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("Median experiments needed (budget = 50)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data = load()
    if not data:
        raise SystemExit("no results in results/bench - run run_benchmark.py first")
    os.makedirs("figures", exist_ok=True)
    rows, tests = summarise(data)
    for fname, strats in GROUPS.items():
        title = {"bench_convergence.png": "BO vs conventional search",
                 "bench_acquisition.png": "Effect of the acquisition function (GP surrogate)",
                 "bench_surrogate.png": "Surrogate model: GP vs neural-network ensemble",
                 "bench_constrained.png": "Constrained BO: best profit that meets the selectivity spec",
                 "bench_log_inputs.png": "GP inputs on a log scale (same search box, different model view)"}[fname]
        convergence_figure(data, strats, os.path.join("figures", fname), title,
                           GROUP_SETTINGS.get(fname))
    final_regret_figure(data, "figures/bench_final_regret.png")
    targets_figure(rows, "figures/bench_evals_to_target.png", GROUPS["bench_convergence.png"])

    for name, recs in [("results/bench_summary.csv", rows), ("results/bench_tests.csv", tests)]:
        with open(name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)

    # curves for the web page: median and IQR per setting x strategy
    curves = {}
    for setting, strats in data.items():
        for s, runs in strats.items():
            _, M = matrix(runs)
            curves.setdefault(setting, {})[s] = dict(
                median=np.median(M, 0).round(6).tolist(),
                q25=np.percentile(M, 25, 0).round(6).tolist(),
                q75=np.percentile(M, 75, 0).round(6).tolist())
    with open("results/bench_summary.json", "w") as fh:
        json.dump(dict(summary=rows, tests=tests, curves=curves,
                       labels={k: v["label"] for k, v in STRATEGIES.items()},
                       settings=SETTING_LABEL), fh)

    # console table
    for setting in SETTING_LABEL:
        rs = sorted([r for r in rows if r["setting"] == setting], key=lambda r: r["median_final_regret"])
        if not rs:
            continue
        print(f"\n== {SETTING_LABEL[setting]}  ({rs[0]['n_seeds']} seeds, {rs[0]['budget']} experiments)")
        print(f"{'strategy':34s}{'median regret':>14s}{'IQR':>20s}{'hit 1%':>8s}{'evals to 1%':>12s}{'time/run':>10s}")
        for r in rs:
            e = r["median_evals_0.01"]
            print(f"{r['label']:34s}{r['median_final_regret']:14.4f}   [{r['q25']:.4f}, {r['q75']:.4f}]"
                  f"{r['success_0.01']:8.0%}{(f'{e:.0f}' if e else '-'):>12s}{r['mean_time_s']:9.1f}s")
    print("\nTests vs GP + EI (Holm-corrected):")
    for t in tests:
        print(f"  {t['setting']:13s} {t['label']:34s} p_holm={t['p_holm']:.3g}  A12={t['A12_ref_better']:.2f}  -> {t['verdict']}")


if __name__ == "__main__":
    main()
