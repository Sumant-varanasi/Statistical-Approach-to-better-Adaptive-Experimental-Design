"""
DEMO FOR PROF. KHARE - why a reactor model that knows the physics beats a black box,
and why it matters more the more complex the chemistry.

Run:   python khare_demo.py                 full demo (first run ~10-15 min on 20 cores,
                                            then cached in results/smart_bench/)
       python khare_demo.py --seeds 5       quicker, fewer repeats
       python khare_demo.py --skip-live     no live campaign (part 3)

It tells the story in four parts, printing as it goes and saving one figure per part:

  PART 1  Two chemistries.  The simple network (the project's A+B->C, A->D) and a complex
          one (A+B<=>C reversible and exothermic, C+B->E over-reaction, A->D cracking).
          How narrow the profitable operating window is for each: the "margin for error".
          -> figures/smart_1_operating_window.png

  PART 2  Head-to-head on both chemistries, repeated over many random starts:
              smart advisor (physics + GP)   vs   plain Bayesian optimisation (v2's best)
              vs   trial and error (one factor at a time)   vs   random search
          plus a stress test where the advisor's model is MISSING the over-reaction.
          -> figures/smart_2_progress.png, figures/smart_3_runs_needed.png

  PART 3  Live: the advisor runs the complex reactor, explaining each choice, then shows
          what it learned about the chemistry (rate constants vs the truth, GC predictions
          vs GC readings).
          -> figures/smart_4_learned_chemistry.png

  PART 4  The bottom line for the person running the reactor, in money and trial runs.
          -> results/smart_demo_summary.md

All numbers use placeholder kinetics and prices (like the rest of the project).  The virtual
reactor has product inhibition and 1 % GC noise that the advisor's model does not know about,
so the advisor's physics is deliberately imperfect, as it would be on a real reactor.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import argparse
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from smart.systems import SYSTEMS, LOW, HIGH, V_I, R, T_REF, unscale, theta_to_physical, NAMES
from smart.advisor import true_optimum, run_smart
from smart import benchmark as B

# colours (validated categorical slots, fixed per method across every figure)
COL = {"gp_ei_logw": "#2a78d6", "smart": "#eb6834", "ofat": "#1baf7a", "random": "#eda100",
       "smart_missing_E": "#e87ba4"}
STYLE = {"gp_ei_logw": "-", "smart": "-", "ofat": "--", "random": ":", "smart_missing_E": "-."}
SYS_COL = {"simple": "#52514e", "complex": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
SHORT = {"smart": "Smart advisor", "gp_ei_logw": "Plain BO", "ofat": "Trial & error",
         "random": "Random", "smart_missing_E": "Smart, wrong network"}
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def say(*a):
    print(*a, flush=True)


def banner(t):
    say("\n" + "=" * 86 + f"\n  {t}\n" + "=" * 86)


# =================================================================================== PART 1
def part1():
    banner("PART 1 - Two chemistries, and how much room for error each one leaves")
    rows = []
    knobs = [3, 4, 0]                       # T, catalyst, pressure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for name, S in SYSTEMS.items():
        o = true_optimum(name)
        x0 = np.array(o["x_opt"])
        best = o["f_opt"]
        say(f"\n  {name.upper()}: {S.description}")
        say(f"    best possible profit {best:.1f} /h at " +
            ", ".join(f"{n.split(' ')[0]}={v:.3g}" for n, v in zip(NAMES, o["x_opt_phys"])))
        say(f"    share of the operating box that makes any profit: {100 * o['frac_profitable']:.1f} %")
        for ax, i in zip(axes, knobs):
            g = np.linspace(0, 1, 241)
            X = np.tile(x0, (len(g), 1))
            X[:, i] = g
            ph = unscale(X)
            P, VA, VB, T, W = ph.T
            F, F0 = S.simulate(P, VA, VB, V_I, W, T, truth=True, method="lsoda")
            rel = 100 * S.profit(F, F0, P, W) / best
            ax.plot(ph[:, i], rel, color=SYS_COL[name], lw=2, label=name)
            ok = ph[rel >= 90, i]
            lo, hi = ok.min(), ok.max()
            frac = (hi - lo) / (HIGH[i] - LOW[i])
            ybar = 120 if name == "simple" else 106
            ax.plot([lo, hi], [ybar, ybar], color=SYS_COL[name], lw=4, solid_capstyle="round")
            unit = NAMES[i].split("(")[1].rstrip(")")
            ax.text((lo + hi) / 2, ybar + 3.5, f"{name}: {hi - lo:.3g} {unit}", ha="center",
                    fontsize=7.5, color=INK2)
            rows.append((name, NAMES[i], lo, hi, frac))
            say(f"    {NAMES[i]:14s} keeps >= 90 % of the best profit only between {lo:7.3g} and {hi:7.3g}"
                f"  ({100 * frac:4.1f} % of the knob's range)")
    for ax, i in zip(axes, knobs):
        style(ax)
        ax.set_xlabel(NAMES[i], fontsize=9, color=INK)
        ax.set_ylim(-60, 132)
        ax.set_yticks([-50, -25, 0, 25, 50, 75, 100])
        ax.axhline(90, color=INK2, lw=0.6, ls=":")
        if i == 4:
            ax.set_xscale("log")
    axes[0].set_ylabel("profit, % of the best possible", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", title="reaction network",
                   title_fontsize=8)
    fig.suptitle("Operating window: profit when one knob moves away from the optimum (others at optimum). "
                 "Bars = range keeping >= 90 % of the best profit.", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig("figures/smart_1_operating_window.png", dpi=150)
    plt.close(fig)
    say("\n  -> figures/smart_1_operating_window.png")
    return rows


# =================================================================================== PART 2
def curves(system, method, seeds):
    H = B.load(system, method, seeds)
    if not H:
        return None
    f = true_optimum(system)["f_opt"]
    return np.array([h["rec"] for h in H]) / f * 100      # % of best possible profit


def runs_to(rel, level=90):
    out = []
    for r in rel:
        hit = np.nonzero(r >= level)[0]
        out.append(hit[0] + 1 if len(hit) else np.inf)
    return np.array(out)


def part2(seeds, budget, workers):
    banner("PART 2 - Head to head: smart advisor vs plain BO vs trial and error vs random")
    say(f"  {seeds} independent campaigns per method and chemistry, {budget} experiments each.")
    say("  Every method sees the same virtual reactor and the same GC noise; profit is always\n"
        "  worked out from the noisy GC reading.  Scored by the TRUE profit of the operating point\n"
        "  each method would recommend after each experiment.\n")
    B.run_all(seeds=seeds, budget=budget, workers=workers)

    methods = list(B.METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    res = {}
    for ax, system in zip(axes, ("simple", "complex")):
        ms = methods + (["smart_missing_E"] if system == "complex" else [])
        ends = []
        for m in ms:
            rel = curves(system, m, seeds)
            if rel is None:
                continue
            res[(system, m)] = rel
            n = np.arange(1, rel.shape[1] + 1)
            med = np.median(rel, 0)
            if m in ("smart", "gp_ei_logw"):
                q1, q3 = np.percentile(rel, [25, 75], 0)
                ax.fill_between(n, np.maximum(q1, -60), q3, color=COL[m], alpha=0.15, lw=0)
            ax.plot(n, np.maximum(med, -60), STYLE[m], color=COL[m], lw=2,
                    label=B.LABELS[m])
            ends.append([float(np.clip(med[-1], -55, 100)), SHORT[m]])
        ends.sort(reverse=True)                       # keep end labels >= 7 points apart,
        ends[0][0] = min(ends[0][0], 94)              # below the 100 % line
        for k in range(1, len(ends)):
            ends[k][0] = min(ends[k][0], ends[k - 1][0] - 7)
        for y, txt in ends:
            ax.text(budget + 0.6, y, txt, fontsize=7.5, color=INK2, va="center")
        ax.axhline(100, color=INK2, lw=0.8, ls=":")
        ax.axvline(8.5, color=GRID, lw=1)
        ax.text(8.8, 103, "<- 8 starting runs | method chooses ->", fontsize=7, color=INK2)
        style(ax)
        ax.set_title(B.SYSTEM_LABEL[system], fontsize=10, color=INK)
        ax.set_xlabel("experiments run on the reactor", fontsize=9, color=INK)
        ax.set_xlim(0.5, budget + 8)
        ax.set_ylim(-62, 110)
    axes[0].set_ylabel("profit of the recommended settings,\n% of the best possible (median over runs)",
                       fontsize=9, color=INK)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Shaded = middle half of the runs (smart advisor and plain BO). Values below -60 % are drawn at -60 %.",
                 fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig("figures/smart_2_progress.png", dpi=150)
    plt.close(fig)

    # table + stats
    lines = []
    for system in ("simple", "complex"):
        f = true_optimum(system)["f_opt"]
        say(f"\n  {B.SYSTEM_LABEL[system]}   (best possible profit {f:.1f} /h)")
        say(f"    {'method':58s} {'after 12':>9s} {'after ' + str(budget):>9s} {'runs to 90%':>12s}")
        for m in methods + (["smart_missing_E"] if system == "complex" else []):
            if (system, m) not in res:
                continue
            rel = res[(system, m)]
            r90 = runs_to(rel)
            med90 = np.median(r90)
            hit = np.mean(np.isfinite(r90))
            s90 = f"{med90:g}" if np.isfinite(med90) else f">{budget}"
            s90 += f" ({100 * hit:.0f}% hit)"
            a12, aN = np.median(rel[:, 11]), np.median(rel[:, -1])
            say(f"    {B.LABELS[m]:58s} {a12:8.0f}% {aN:8.0f}% {s90:>16s}")
            lines.append((system, m, a12, aN, med90, hit))
        a, b_ = res[(system, "smart")], res[(system, "gp_ei_logw")]
        k = min(len(a), len(b_))
        for at in (11, budget - 1):
            try:
                p = wilcoxon(a[:k, at], b_[:k, at]).pvalue
            except ValueError:
                p = 1.0
            say(f"    smart vs plain BO after {at + 1:2d} experiments: smart better in "
                f"{np.sum(a[:k, at] > b_[:k, at])}/{k} paired runs, Wilcoxon p = {p:.3g}")
    plot_runs_needed(res, budget)
    say("\n  -> figures/smart_2_progress.png, figures/smart_3_runs_needed.png")
    return res, lines


def plot_runs_needed(res, budget):
    methods = ["smart", "gp_ei_logw", "ofat", "random"]
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    w = 0.19
    for j, m in enumerate(methods):
        for i, system in enumerate(("simple", "complex")):
            rel = res.get((system, m))
            if rel is None:
                continue
            r90 = runs_to(rel)
            med = np.median(r90)
            x = i + (j - 1.5) * (w + 0.02)
            if np.isfinite(med):
                ax.bar(x, med, w, color=COL[m], label=B.LABELS[m] if i == 0 else None)
                ax.text(x, med + 0.6, f"{med:g}", ha="center", fontsize=8, color=INK)
            else:
                ax.bar(x, budget, w, color="none", edgecolor=COL[m], hatch="///", lw=1,
                       label=B.LABELS[m] if i == 0 else None)
                ax.text(x, budget + 0.6, f">{budget}", ha="center", fontsize=8, color=INK)
            hit = np.mean(np.isfinite(r90))
            ax.text(x, -2.2, f"{100 * hit:.0f}%", ha="center", fontsize=7, color=INK2)
    style(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([B.SYSTEM_LABEL["simple"], B.SYSTEM_LABEL["complex"]], fontsize=9)
    ax.set_ylim(-4, budget + 5)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= budget])
    ax.set_ylabel("experiments needed (median)", fontsize=9, color=INK)
    ax.set_title("Experiments needed to reach 90 % of the best possible profit\n"
                 "(hatched = not reached within the budget; small % = share of runs that got there)",
                 fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.tight_layout()
    fig.savefig("figures/smart_3_runs_needed.png", dpi=150)
    plt.close(fig)


# =================================================================================== PART 3
def part3(budget_live=16, seed=0):
    banner("PART 3 - Live: the advisor runs the complex reactor and explains itself")
    S = SYSTEMS["complex"]
    f = true_optimum("complex")["f_opt"]
    say(f"  8 starting experiments spread over the whole box, then the advisor chooses.\n"
        f"  (best possible profit on this reactor: {f:.1f} /h - the advisor does not know this)\n")
    say("  run   P(bar) V_A  V_B   T(K)  Wcat(g)   expected      measured  true   why")
    t0 = time.time()

    def cb(n, x, r, info):
        P, VA, VB, T, W = unscale(x)[0]
        why = "model sure it is better" if info["pred"] - info["best_pred"] > 1.64 * info["sd"] \
            else "model unsure here: checking"
        say(f"  {n:3d}  {P:6.0f} {VA:4.0f} {VB:4.0f} {T:6.0f} {W:7.2f}   {info['pred']:5.1f}±{info['sd']:4.1f}"
            f"   {r['profit']:8.1f} {r['profit_true']:6.1f}   {why}")

    h = run_smart("complex", budget=budget_live, seed=seed, callback=cb)
    adv = h["advisor"]
    say(f"\n  ({time.time() - t0:.0f} s)  Recommended settings now make {h['rec'][-1]:.1f} /h "
        f"= {100 * h['rec'][-1] / f:.0f} % of the best possible.")

    th_fit, th_true = adv.kin.theta, S.theta_true
    k_f, E_f = theta_to_physical(th_fit)
    k_t, E_t = theta_to_physical(th_true)
    sd = adv.kin.param_sd()[1::2] * R * T_REF / 1000
    names = ["A+B<=>C (main)", "C+B->E (over-reaction)", "A->D (cracking)"]
    say("\n  What it learned (from GC readings only):")
    for j in range(3):
        say(f"    {names[j]:24s} Ea fitted {E_f[j]:5.0f} ± {sd[j]:3.0f} kJ/mol   true {E_t[j]:4.0f}"
            f"    k_ref fitted/true = {k_f[j] / k_t[j]:.2f}")
    say(f"    (rate constants come out about {1 / np.median(k_f / k_t):.1f}x lower than the truth: the real reactor\n"
        "     has product inhibition that slows every reaction and is not in the model, so the fitted\n"
        "     constants absorb it. The activation energies, which decide the best temperature, are\n"
        "     close, and the GP correction covers what is left.)")

    plot_learned(adv, th_true, th_fit, names)
    say("\n  -> figures/smart_4_learned_chemistry.png")
    return h


def plot_learned(adv, th_true, th_fit, names):
    """Left: Arrhenius plot of learned vs true rate constants (90 % band from the fit's
    uncertainty).  Right: every GC number vs the learned model's prediction."""
    S = SYSTEMS["complex"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    Tg = np.linspace(450, 850, 100)
    draws = adv.kin.samples(200, np.random.default_rng(0))
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for j in range(3):
        lg = lambda th: (th[2 * j] - th[2 * j + 1] * (T_REF / Tg - 1)) / np.log(10)
        band = np.array([lg(d) for d in draws])
        a1.fill_between(1000 / Tg, *np.percentile(band, [5, 95], 0), color=cols[j], alpha=0.18, lw=0)
        a1.plot(1000 / Tg, lg(th_true), color=cols[j], lw=2, label=f"{names[j]}: truth")
        a1.plot(1000 / Tg, lg(th_fit), color=cols[j], lw=1.6, ls="--", label=f"{names[j]}: learned")
    style(a1)
    a1.set_xlabel("1000 / T  (1/K)      <- hotter", fontsize=9, color=INK)
    a1.set_ylabel("log10 rate constant k(T)", fontsize=9, color=INK)
    a1.set_title("Arrhenius plot: learned (dashed, 90 % band) vs true (solid)", fontsize=9.5, color=INK)
    a1.legend(frameon=False, fontsize=7, loc="lower left")
    pred = adv.kin.predict_gc(unscale(adv.X))
    meas = adv.gc
    for k, sp in enumerate(S.species[:-1]):
        c = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][k]
        a2.scatter(meas[:, k], pred[:, k], s=24, color=c, edgecolor="white", lw=0.6, label=sp, zorder=3)
    lim = [0, max(meas[:, :-1].max(), pred[:, :-1].max()) * 1.05]
    a2.plot(lim, lim, color=INK2, lw=0.8, ls=":")
    style(a2)
    a2.set_xlabel("GC reading (mole fraction)", fontsize=9, color=INK)
    a2.set_ylabel("model prediction", fontsize=9, color=INK)
    a2.set_title(f"All GC numbers from {len(meas)} runs vs the learned model", fontsize=9.5, color=INK)
    a2.legend(frameon=False, fontsize=8, title="species", title_fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/smart_4_learned_chemistry.png", dpi=150)
    plt.close(fig)


# =================================================================================== PART 4
def part4(windows, lines, budget, seeds):
    banner("PART 4 - The bottom line for the person running the reactor")
    get = {(s, m): (a12, aN, r90, hit) for s, m, a12, aN, r90, hit in lines}
    out = ["# Smart reactor advisor - demo summary", "",
           f"{seeds} campaigns per method and chemistry, {budget} experiments each. Placeholder kinetics and prices.",
           "Profit = true profit of the settings each method recommends, as % of the best possible.", "",
           "| Chemistry | Method | after 12 experiments | after " + str(budget) + " | experiments to reach 90 % |",
           "|---|---|---|---|---|"]
    for (s, m), (a12, aN, r90, hit) in get.items():
        r = f"{r90:g}" if np.isfinite(r90) else f">{budget}"
        out.append(f"| {s} | {B.LABELS[m]} | {a12:.0f} % | {aN:.0f} % | {r} ({100 * hit:.0f} % of runs got there) |")
    out += ["", "## Operating window (range of each knob that keeps >= 90 % of the best profit)", "",
            "| Chemistry | Knob | from | to | share of range |", "|---|---|---|---|---|"]
    for name, knob, lo, hi, frac in windows:
        out.append(f"| {name} | {knob} | {lo:.3g} | {hi:.3g} | {100 * frac:.0f} % |")

    for system in ("simple", "complex"):
        s12, b12, o12 = (get[(system, m)][0] for m in ("smart", "gp_ei_logw", "ofat"))
        f = true_optimum(system)["f_opt"]
        say(f"\n  {system.upper()} reaction, after 12 trial runs (8 starting runs + 4 chosen):")
        say(f"    smart advisor  : {s12:4.0f} % of the best possible profit  ({s12 * f / 100:5.1f} /h)")
        say(f"    plain BO       : {b12:4.0f} %                              ({b12 * f / 100:5.1f} /h)")
        say(f"    trial and error: {o12:4.0f} %                              ({o12 * f / 100:5.1f} /h)")
    sr, br = get[("simple", "smart")][2], get[("simple", "gp_ei_logw")][2]
    cr, cb = get[("complex", "smart")][2], get[("complex", "gp_ei_logw")][2]
    fmt = lambda v: f"{v:g}" if np.isfinite(v) else f"more than {budget}"
    say(f"\n  Runs needed to reach 90 % of the best profit (median):")
    say(f"    simple : smart {fmt(sr)},  plain BO {fmt(br)}")
    say(f"    complex: smart {fmt(cr)},  plain BO {fmt(cb)}")
    gap = {s: get[(s, "smart")][1] - get[(s, "gp_ei_logw")][1] for s in ("simple", "complex")}
    reading = (
        f"\n  Reading: after {budget} runs plain BO reaches {get[('simple', 'gp_ei_logw')][1]:.0f} % on the simple "
        f"reaction but only\n  {get[('complex', 'gp_ei_logw')][1]:.0f} % on the complex one; the smart advisor reaches "
        f"{get[('simple', 'smart')][1]:.0f} % and {get[('complex', 'smart')][1]:.0f} %.\n"
        f"  The advantage of knowing the physics is {gap['simple']:.0f} points on the simple reaction and "
        f"{gap['complex']:.0f} points on the\n  complex one. On a simple reaction the profitable region is wider "
        "and a black box\n  eventually finds most of it: there is less to gain. On the complex reaction the "
        "window is\n  narrow and shaped by competing reactions; a black box that only sees profit keeps missing\n"
        "  it, while the advisor reads the full GC composition, learns each reaction's temperature\n"
        "  response and the equilibrium limit, and goes straight to the window.")
    say(reading)
    out += ["", "## Reading", "", " ".join(reading.split())]
    with open("results/smart_demo_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    say("\n  -> results/smart_demo_summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-live", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    windows = part1()
    res, lines = part2(a.seeds, a.budget, a.workers)
    if not a.skip_live:
        part3()
    part4(windows, lines, a.budget, a.seeds)
    say(f"\nDone in {time.time() - t0:.0f} s.  Try the operator's tool:  python reactor_advisor.py --demo")


if __name__ == "__main__":
    main()
