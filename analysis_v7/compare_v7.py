"""
Final v7 comparison: the full V7 study (benchmark.V7_STUDY, 18 methods x 10 seeds x budget 25) next
to the committed v5/v6 benchmark.  Run from the repository root:

    python analysis_v7/compare_v7.py

Reads campaign files from (first found wins, stale definitions skipped, see benchmark.is_stale)
    results/smart_bench            v5, v6, baselines (the committed benchmark cache)
    results/hpc_v7final/bench      the V7 study run on the HPC (seeds 0-6)
    results/v7final_local/bench    the V7 study run on the laptop (seeds 7-9), same code
Writes
    figures/smart_7_campaigns.png    all advisors and the neural network alone, every scenario;
                                     runs to 90 %; what each part adds; held-out prediction error
    figures/smart_7_candidates.png   starting from a wrong network (close / similar / far): v5 and
                                     v7 with the network fixed against v7 that may repair it, and
                                     which network v7 ended with
    analysis_v7/v7_summary.md            the numbers behind both, with paired tests (same seed = same
                                     screening runs and the same reactor noise)
"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from smart.advisor import true_optimum
from smart.systems import SYSTEMS
from smart import benchmark as B
from smart import discovery as D

ROOTS = ["results/smart_bench", "results/hpc_v7final/bench", "results/v7final_local/bench"]
SKIPPED = set()
SEEDS, BUDGET = 10, 25
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8986", "#e6e5e1", "#fcfcfb"
COL = {"nn_only": "#eb6834", "plain_bo": "#e87ba4", "chemist": "#eda100", "smart": "#2a78d6",
       "smart_v6": "#1baf7a", "smart_v7": "#008300"}
NAME = {"nn_only": "Neural network alone", "plain_bo": "Plain BO (GP, profit only)", "chemist": "Rules of thumb (script)",
        "smart": "v5 advisor (physics + GP)", "smart_v6": "v6 advisor (physics + GP + NNs)",
        "smart_v7": "v7 advisor (v6 + NN alone + repair)"}
ORDER = ["nn_only", "plain_bo", "chemist", "smart", "smart_v6", "smart_v7"]
SCEN = [("simple", "", "Simple reaction"), ("complex", "", "Complex reaction, model correct"),
        ("complex", "_missing_E", "Complex, C+B→E missing from the model"),
        ("complex_deact", "", "Complex, catalyst ageing")]
MODEL_FREE = {"nn_only", "plain_bo", "chemist"}
_F = {}


def load(st, m, k):
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
    """{seed: campaign} of one method; f_opt of the setting."""
    if st not in _F:
        _F[st] = true_optimum(B.SETTINGS[st]["system"])["f_opt"]
    out = {k: load(st, m, k) for k in range(SEEDS)}
    return _F[st], {k: d for k, d in out.items() if d is not None}


def runs_to(d, f, level=90.0):
    hit = np.nonzero(100 * d["rec"] / f >= level)[0]
    return hit[0] + 1 if len(hit) else np.inf


def stats(st, m):
    f, C = campaigns(st, m)
    if not C:
        return None
    final = np.array([100 * d["rec"][-1] / f for d in C.values()])
    earned = np.array([d["cum_true"][-1] for d in C.values()])
    t90 = np.array([runs_to(d, f) for d in C.values()], float)
    secs = np.array([float(d["time_total"]) if "time_total" in d.files else np.nan for d in C.values()])
    return dict(n=len(C), final=float(np.median(final)), final_lo=float(np.percentile(final, 25)),
                earned=float(np.median(earned)), earned_q=(float(np.percentile(earned, 25)), float(np.percentile(earned, 75))),
                t90=float(np.median(t90)), reached=int(np.isfinite(t90).sum()), secs=float(np.nanmedian(secs)))


def paired(st, a, b, key="earned"):
    """Paired difference a - b over the seeds both have (median, wins of a, Wilcoxon p)."""
    f, A = campaigns(st, a)
    _, Bb = campaigns(st, b)
    ks = sorted(set(A) & set(Bb))
    if len(ks) < 3:
        return None
    val = (lambda d: d["cum_true"][-1]) if key == "earned" else (lambda d: 100 * d["rec"][-1] / f)
    diff = np.array([val(A[k]) - val(Bb[k]) for k in ks])
    p = float(wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-9) else 1.0
    return dict(n=len(ks), median=float(np.median(diff)), wins=int(np.sum(diff > 1e-9)), losses=int(np.sum(diff < -1e-9)), p=p)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =================================================================================== figure 1: campaigns
def figure_campaigns():
    fig = plt.figure(figsize=(19, 10.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.42, wspace=0.28)
    for j, (st, suf, title) in enumerate(SCEN):
        ax = fig.add_subplot(gs[0, j])
        style(ax)
        ax.axvspan(0.5, 5.5, color="#efeeea", lw=0)
        ax.axhline(100, color=MUTED, lw=0.8, ls=":")
        for m in ORDER:
            f, C = campaigns(st, method_name(m, suf))
            if not C:
                continue
            R = np.array([100 * d["rec"] / f for d in C.values()])
            med = np.maximum(np.median(R, 0), -60)
            ax.plot(np.arange(1, R.shape[1] + 1), med, color=COL[m], lw=2.2 if m.startswith("smart") else 1.6,
                    label=NAME[m], solid_capstyle="round", zorder=3 if m == "smart_v7" else 2)
        ax.set_ylim(-63, 112)
        ax.set_xlim(0.5, BUDGET + 0.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("runs on the reactor", fontsize=8, color=INK2)
        if j == 0:
            ax.set_ylabel("profit of the recommended settings\n(% of the best possible, median of 10)", fontsize=8, color=INK2)
            ax.text(3, -58, "5 shared\nscreening runs", fontsize=7, color=INK2, ha="center")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.975))

    # ---- runs to 90 %
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor(SURF)
    ax.axis("off")
    cols = ["Simple reaction", "Complex,\nmodel correct", "Complex,\nC+B→E missing", "Complex,\ncatalyst ageing"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#1c5cab", "#86b6ef", "#e8f1fc"])
    ax.text(0.0, 1.03, "Runs needed to reach 90 % of the best profit", fontsize=10, color=INK, transform=ax.transAxes)
    ax.text(0.0, 0.975, "median of 10 campaigns, fewer is better; small print = campaigns that got there within 25 runs",
            fontsize=7.5, color=INK2, transform=ax.transAxes)
    x0, cw, y0, rh = 0.355, 0.16, 0.83, 0.125
    for j, c in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cw, y0 + 0.07, c, ha="center", va="center", fontsize=8, color=INK2, transform=ax.transAxes)
    for i, m in enumerate(ORDER):
        yy = y0 - (i + 1) * rh
        ax.add_patch(plt.Rectangle((0.0, yy + rh * 0.3), 0.012, rh * 0.4, color=COL[m], transform=ax.transAxes, lw=0))
        ax.text(0.02, yy + rh / 2, NAME[m], fontsize=8.3, color=INK, va="center", transform=ax.transAxes)
        for j, (st, suf, _) in enumerate(SCEN):
            s = stats(st, method_name(m, suf))
            if s is None:
                ax.text(x0 + (j + 0.5) * cw, yy + rh / 2, "no data", ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
                continue
            med = s["t90"]
            if np.isfinite(med):
                fc, big, tc = cmap(float(min(1.0, max(0.0, (med - 8.0) / 14.0)))), f"{med:g}", ("#ffffff" if med < 13 else INK)
            else:
                fc, big, tc = "#f1f0ec", "never", INK2
            ax.add_patch(plt.Rectangle((x0 + j * cw + 0.003, yy + 0.008), cw - 0.006, rh - 0.016, color=fc, transform=ax.transAxes, lw=0))
            ax.text(x0 + (j + 0.5) * cw, yy + rh * 0.6, big, ha="center", va="center", fontsize=10, color=tc,
                    transform=ax.transAxes, fontweight="bold" if np.isfinite(med) else "normal")
            ax.text(x0 + (j + 0.5) * cw, yy + rh * 0.25, f"{s['reached']}/{s['n']}", ha="center", va="center", fontsize=7,
                    color=tc, transform=ax.transAxes)

    # ---- what each part adds (C+B->E missing)
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    parts = [("smart_missing_E", "v5: physics + GP", "#2a78d6"),
             ("smart_v6_nonn_missing_E", "v6 without its NNs", "#86b6ef"),
             ("smart_v6_nogp_missing_E", "v6 without its GP", "#8fd5bb"),
             ("smart_v6_missing_E", "v6: physics + GP + NNs", "#1baf7a"),
             ("smart_v7_norevise_missing_E", "v6 + NN alone (v7, no repair)", "#66b266"),
             ("smart_v7_missing_E", "v7: + network repair", "#008300")]
    lo_x = 0.0
    for i, (m, lab, c) in enumerate(parts):
        _, C = campaigns("complex", m)
        y = len(parts) - 1 - i
        if not C:
            ax.text(5, y, f"{lab}: no data", fontsize=7.5, color=MUTED, va="center")
            continue
        E = np.array([d["cum_true"][-1] for d in C.values()])
        q1, med, q3 = np.percentile(E, [25, 50, 75])
        lo_x = min(lo_x, q1)
        ax.barh(y, med, height=0.42, color=c)
        ax.plot([q1, q3], [y, y], color=INK, lw=1.1)
        ax.text(-125, y + 0.36, lab, fontsize=7.8, color=INK, va="center")
        ax.text(max(med, q3) + 5, y, f"{med:.0f}", va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(-128, 300)
    ax.set_ylim(-0.5, len(parts) - 0.2)
    ax.set_title("What each part adds, C+B→E missing", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("profit earned over the 25-run campaign\n(median of 10; black line = middle half of the campaigns)", fontsize=8, color=INK2)

    # ---- held-out prediction error of the committee members (v7 campaigns)
    ax = fig.add_subplot(gs[1, 3])
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    mem = [("none", "physics alone", MUTED), ("gp", "physics + GP", "#2a78d6"),
           ("nn", "physics + NNs", "#1baf7a"), ("direct", "neural network alone", "#eb6834")]
    y, xmax = 0, 10
    for m, title in [("smart_v7_norevise_missing_E", "C+B→E missing, no repair"), ("smart_v7", "Model correct")]:
        _, C = campaigns("complex", m)
        if not C:
            continue
        E = np.array([d["committee_heldout_rmse"] for d in C.values()])
        cols4 = list(list(C.values())[0]["weight_columns"])
        for key, lab, c in mem[::-1]:
            v = np.nanmean(E[:, -5:, cols4.index(key)], 1)
            q1, med, q3 = np.nanpercentile(v, [25, 50, 75])
            xmax = max(xmax, q3)
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
    ax.set_xlim(0, xmax * 1.45 + 2)
    ax.set_title("Error predicting runs it has not seen (v7)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("typical error of the predicted profit (money/h)\nheld-out runs after 20–24 runs, median of 10", fontsize=8, color=INK2)

    fig.text(0.01, 0.003, "Simulated reactor, placeholder kinetics and prices; 10 seeds x 25 runs per method (v7 study run on the HPC "
             "and the laptop with the same code; v5/v6 and baselines from the committed benchmark). Curves clipped at -60 %. "
             "NN = neural network, GP = Gaussian process.", fontsize=7.5, color=INK2)
    out = "figures/smart_7_campaigns.png"
    fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return out


# =================================================================================== figure 2: candidate networks
TRUTH_KEY = D.network_key(D.parse_network(D.reactions_from_system(SYSTEMS["complex"])))
TRUE_KEYS = {r.key() for r in D.reactions_from_system(SYSTEMS["complex"])}
STARTS = [("true", "the true network", "smart", "smart_v7_fixednet", "smart_v7"),
          ("close", "quite close\n(one order wrong)", "smart_cand_close", "smart_v7_fixednet_cand_close", "smart_v7_cand_close"),
          ("similar", "relatively the same\n(a different pathway)", "smart_cand_similar", "smart_v7_fixednet_cand_similar", "smart_v7_cand_similar"),
          ("far", "very different\n(two reactions, no E)", "smart_cand_far", "smart_v7_fixednet_cand_far", "smart_v7_cand_far")]
TRIO = [("v5, network fixed", "#2a78d6"), ("v7, network fixed", "#8fd5bb"), ("v7, may repair it", "#008300")]


def outcome(d):
    """'true' / 'true + extra' / 'other' for the network a campaign ended with."""
    txt = str(d["network_final"]) if "network_final" in d.files else None
    if txt is None:
        return None
    net = D.parse_network(txt)
    if D.network_key(net) == TRUTH_KEY:
        return "true"
    keys = {r.key() for r in net}
    return "true + extra" if TRUE_KEYS <= keys else "other"


def figure_candidates():
    fig = plt.figure(figsize=(17, 6.4), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=0.32)
    for p, (metric, title, xlab) in enumerate([
            ("earned", "Profit earned over the 25-run campaign", "money/h summed over the runs (median of 10; line = middle half)"),
            ("final", "Where it ends: profit of the recommended settings", "% of the best possible after 25 runs (median of 10; line = middle half)")]):
        ax = fig.add_subplot(gs[0, p])
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        yt, yl = [], []
        for i, (key, lab, *ms) in enumerate(STARTS):
            base = (len(STARTS) - 1 - i) * 4.0
            for k, (m, (tl, c)) in enumerate(zip(ms, TRIO)):
                f, C = campaigns("complex", m)
                y = base + 2 - k
                if not C:
                    ax.text(1, y, "no data", fontsize=7, color=MUTED, va="center")
                    continue
                v = np.array([d["cum_true"][-1] if metric == "earned" else 100 * d["rec"][-1] / f for d in C.values()])
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.barh(y, med, height=0.8, color=c, label=tl if i == 0 else None)
                ax.plot([q1, q3], [y, y], color=INK, lw=1.0)
                gap = 4 if metric == "earned" else 1.5
                if med < 0:
                    ax.text(min(med, q1) - gap, y, f"{med:.0f}", va="center", ha="right", fontsize=7.5, color=INK)
                else:
                    ax.text(max(med, q3) + gap, y, f"{med:.0f}", va="center", fontsize=7.5, color=INK)
            yt.append(base + 1)
            yl.append(lab)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl if p == 0 else [""] * len(yl), fontsize=8.5, color=INK)
        ax.axvline(0, color=MUTED, lw=0.8)
        if metric == "earned":
            ax.set_xlim(ax.get_xlim()[0] - 40, ax.get_xlim()[1])
        if metric == "final":
            ax.axvline(100, color=MUTED, lw=0.8, ls=":")
            ax.set_xlim(min(-5, ax.get_xlim()[0]), 118)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=30)
        ax.set_xlabel(xlab, fontsize=8, color=INK2)
        if p == 0:
            ax.set_ylabel("network the advisor is given at the start", fontsize=8.5, color=INK2)
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    # ---- which network v7 ended with
    ax = fig.add_subplot(gs[0, 2])
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    cat = [("true", "the true network", "#008300"), ("true + extra", "true + an extra reaction", "#eda100"),
           ("other", "a wrong network", "#e34948")]
    for i, (key, lab, *ms) in enumerate(STARTS):
        _, C = campaigns("complex", ms[2])
        y = len(STARTS) - 1 - i
        got = [outcome(d) for d in C.values()]
        left = 0
        for ck, cl, cc in cat:
            n = sum(g == ck for g in got)
            if n:
                ax.barh(y, n, left=left, height=0.55, color=cc, label=cl if cl not in ax.get_legend_handles_labels()[1] else None,
                        edgecolor=SURF, lw=1.5)
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8.5, color="#ffffff", fontweight="bold")
            left += n
        ax.text(-0.3, y, STARTS[i][1].split("\n")[0], ha="right", va="center", fontsize=8.3, color=INK)
    ax.set_yticks([])
    ax.set_xlim(-6.5, SEEDS + 0.3)
    ax.set_xticks(range(0, SEEDS + 1, 2))
    ax.set_title("v7: which network it ended with", fontsize=10, color=INK, loc="left", pad=30)
    ax.set_xlabel("campaigns out of 10 (complex reaction, 25 runs)", fontsize=8, color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=8.0)
    fig.text(0.01, -0.03, "Complex reaction (true network A + B ⇌ C, C + B → E, A → D), simulated reactor with GC noise, placeholder "
             "kinetics and prices. 'v7, may repair it' changes its reactions only on strong evidence (BIC gain of 10), one change "
             "at a time; it never removes a reaction that does no harm.", fontsize=7.5, color=INK2)
    out = "figures/smart_7_candidates.png"
    fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return out


# =================================================================================== the summary table
def machine_check():
    """Campaigns run on both the laptop and the HPC (same code, same seed): how far apart they ended."""
    import glob
    lap, hpc = "results/v7final_local/bench", "results/hpc_v7final/bench"
    both, diverged = 0, []
    for p in sorted(glob.glob(os.path.join(lap, "*", "*.npz"))):
        q = os.path.join(hpc, os.path.relpath(p, lap))
        if not os.path.exists(q):
            continue
        a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        if np.array_equal(a["X"], b["X"]) and np.array_equal(a["y_obs"], b["y_obs"]):
            continue                                   # a copy (laptop results were uploaded to the HPC run)
        both += 1
        dx = np.any(np.abs(a["X"] - b["X"]) > 1e-6, axis=1)
        if dx.any():
            diverged.append((os.path.relpath(p, lap), int(np.nonzero(dx)[0][0]) + 1, float(a["cum_true"][-1] - b["cum_true"][-1])))
    if not both:
        return ["No campaign was run on both machines."]
    d = np.abs([x[2] for x in diverged]) if diverged else np.zeros(1)
    out = [f"{both} campaigns were run on both the laptop and the HPC with the same code and seed. In {both - len(diverged)} "
           f"the settings chosen agree to 1e-6 throughout (only the last digits of the numbers differ); in {len(diverged)} the "
           f"two machines' rounding tipped a close decision and the campaigns went different ways from some run on "
           f"(median difference in profit earned {np.median(d):.0f}, largest {np.max(d):.0f}):", ""]
    out += [f"- {name.replace(os.sep, '/')}: from run {r}, earned {e:+.0f} (laptop minus HPC)" for name, r, e in diverged]
    out += ["", "A campaign is reproducible on one machine; across machines, campaigns whose model is wrong can diverge. "
            "The same seed still gives every method the same screening runs and the same reactor noise, so the pairing "
            "holds. Each campaign is used once (the HPC copy where there are two)."]
    return out


def fmt_row(st, m, label):
    s = stats(st, m)
    if s is None:
        return f"| {label} | – | – | – | – | – |"
    t = "never" if not np.isfinite(s["t90"]) else f"{s['t90']:g}"
    return (f"| {label} | {s['n']} | {s['final']:.1f} | {s['earned']:.0f} ({s['earned_q'][0]:.0f} to {s['earned_q'][1]:.0f}) | "
            f"{t} ({s['reached']}/{s['n']}) | {s['secs'] / 60:.1f} |")


def fmt_pair(st, a, b, label):
    r = paired(st, a, b)
    if r is None:
        return f"| {label} | – | – | – |"
    return f"| {label} | {r['median']:+.0f} | {r['wins']} / {r['losses']} (of {r['n']}) | {r['p']:.3f} |"


def summary():
    L = ["# v7 study: results", "",
         "Simulated reactor, placeholder kinetics and prices. 10 seeds x 25 runs per method; the same seed gives every method "
         "the same 5 screening runs and the same reactor noise, so differences are paired. Generated by "
         "`python analysis_v7/compare_v7.py`.", "",
         "Columns: campaigns; final = profit of the recommended settings after 25 runs, % of the best possible (median); "
         "earned = profit earned over the 25 runs (median, middle half in brackets); runs to 90 % (median; campaigns that "
         "got there); minutes per campaign (median, wall time on the machine that ran it).", ""]
    rows = [("nn_only", "Neural network alone"), ("plain_bo", "Plain BO"), ("chemist", "Rules of thumb"),
            ("smart", "v5"), ("smart_v6", "v6"), ("smart_v6_nonn", "v6 without NNs"), ("smart_v6_nogp", "v6 without GP"),
            ("smart_v7_fixednet", "v7, network fixed"), ("smart_v7", "v7")]
    for st, suf, title in SCEN:
        L += [f"## {title}", "", "| method | n | final % | earned | runs to 90 % | min |", "|---|---|---|---|---|---|"]
        for m, lab in rows:
            if suf and m in ("smart_v7_fixednet",):
                m2 = "smart_v7_norevise_missing_E"
                lab = "v7, network fixed"
            else:
                m2 = method_name(m, suf)
            L.append(fmt_row(st, m2, lab))
        L.append("")
    L += ["## Paired differences in profit earned (first minus second; Wilcoxon signed-rank test)", "",
          "| comparison | median difference | first better / worse | p |", "|---|---|---|---|"]
    for st, suf, title in SCEN:
        L.append(f"| **{title}** | | | |")
        sx = lambda m: method_name(m, suf)
        L.append(fmt_pair(st, sx("smart_v7"), sx("smart_v6"), "v7 vs v6"))
        L.append(fmt_pair(st, sx("smart_v7"), "smart_v7_norevise_missing_E" if suf else "smart_v7_fixednet", "v7 vs v7 with the network fixed"))
        L.append(fmt_pair(st, "smart_v7_norevise_missing_E" if suf else "smart_v7_fixednet", sx("smart_v6"), "v6 + NN alone (no repair) vs v6"))
        L.append(fmt_pair(st, sx("smart_v6_nonn"), sx("smart_v6"), "v6 without NNs vs v6"))
        L.append(fmt_pair(st, sx("smart_v6_nogp"), sx("smart_v6"), "v6 without GP vs v6"))
        L.append(fmt_pair(st, "nn_only", sx("smart_v6"), "neural network alone vs v6"))
    L += ["", "## Starting from a wrong network (complex reaction)", "",
          "| given network | method | n | final % | earned | runs to 90 % | min | ended with (true / true + extra / other) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, lab, *ms in STARTS:
        for m, (tl, _) in zip(ms, TRIO):
            r = fmt_row("complex", m, "x").split("|")[2:-1]
            _, C = campaigns("complex", m)
            oc = [outcome(d) for d in C.values()]
            ended = (f"{oc.count('true')} / {oc.count('true + extra')} / {oc.count('other')}"
                     if oc and all(o is not None for o in oc) and m.startswith("smart_v7") and "fixednet" not in m else "(fixed)")
            L.append(f"| {lab.splitlines()[0]} | {tl} |" + "|".join(r) + f"| {ended} |")
    L += ["", "| given network | v7 repair vs v7 fixed: median difference | better / worse | p |", "|---|---|---|---|"]
    for key, lab, *ms in STARTS:
        r = paired("complex", ms[2], ms[1])
        if r:
            L.append(f"| {lab.splitlines()[0]} | {r['median']:+.0f} | {r['wins']} / {r['losses']} (of {r['n']}) | {r['p']:.3f} |")
    # when the missing reaction was put back
    _, C = campaigns("complex", "smart_v7_missing_E")
    first = []
    for d in C.values():
        found = None
        for j, txt in enumerate(d["network"]):
            keys = {r.key() for r in D.parse_network(str(txt))}
            if TRUE_KEYS <= keys:
                found = j + 1
                break
        first.append(found)
    got = [f for f in first if f is not None]
    if C:
        L += ["", f"C+B→E missing at the start: v7 had all three true reactions by run {int(np.median(got)) if got else '–'} "
                  f"(median) in {len(got)} of {len(first)} campaigns; it ended with the true network in "
                  f"{sum(outcome(d) == 'true' for d in C.values())} of them."]
    L += ["", "## Reproducibility across machines", ""] + machine_check()
    if SKIPPED:
        L += ["", f"{len(SKIPPED)} cached file(s) made by an earlier definition of their method were skipped."]
    with open("analysis_v7/v7_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return "analysis_v7/v7_summary.md"


if __name__ == "__main__":
    for out in (figure_campaigns(), figure_candidates(), summary()):
        print("->", out)
