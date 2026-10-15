"""
DEMO FOR PROF. KHARE - a reactor model that knows the physics, run the way the reactor is
really run, against what an experienced chemist would do by hand.

Run:   python khare_demo.py                 full demo (first run ~10 min on 20 cores,
                                            then cached in results/smart_bench/)
       python khare_demo.py --seeds 5       quicker, fewer repeats
       python khare_demo.py --skip-live     no live campaign (part 3)

How every campaign is run (the realistic protocol, same for every method):
  * the catalyst (1 g, 100 m2/g) is loaded once and never changed
  * the operator's 5 screening runs come first: a rising temperature ladder, then pressure
    up, then more B - identical for every method
  * from run 6 on, each method proposes the next run; it must be within the step limits of
    the previous run (10 bar, 10 mL/min per flow, 30 K), inside the box, and leave enough
    total flow for the GC; 25 runs in all (five weeks at one run per day)
  * five knobs: pressure, the flows of A, B and the inert, temperature

The story in four parts, printing as it goes and saving one figure per part:

  PART 1  Two chemistries.  The simple network (the project's A+B->C, A->D) and a complex
          one (A+B<=>C reversible and exothermic, C+B->E over-reaction, A->D cracking).
          How narrow the profitable operating window is for each: the "margin for error".
          -> figures/smart_1_operating_window.png

  PART 2  Head-to-head on both chemistries, repeated over 10 starting points:
              smart advisor (physics + GP)   vs   a chemist's rules of thumb (hill climbing
              with chemistry heuristics)   vs   plain Bayesian optimisation   vs   random walk
          plus a stress test where the advisor's model is MISSING the over-reaction.
          -> figures/smart_2_progress.png, figures/smart_3_runs_needed.png

  PART 3  Live: the advisor runs the complex reactor, explaining each choice, then shows
          what it learned about the chemistry (rate constants vs the truth, GC predictions
          vs GC readings).
          -> figures/smart_4_learned_chemistry.png

  PART 4  The bottom line for the person running the reactor, in money and runs.
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

from smart.systems import (SYSTEMS, LOW, HIGH, MAX_STEP, WCAT, A_SPEC, R, T_REF, NAMES, unscale,
                           theta_to_physical, physical)
from smart.advisor import true_optimum, run_smart
from smart import benchmark as B

# colours (validated categorical slots, fixed per method across every figure)
COL = {"plain_bo": "#2a78d6", "smart": "#eb6834", "chemist": "#1baf7a", "random_walk": "#eda100",
       "smart_missing_E": "#e87ba4"}
STYLE = {"plain_bo": "-", "smart": "-", "chemist": "--", "random_walk": ":", "smart_missing_E": "-."}
SYS_COL = {"simple": "#52514e", "complex": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
SHORT = {"smart": "Smart advisor", "plain_bo": "Plain BO", "chemist": "Chemist's rules",
         "random_walk": "Random walk", "smart_missing_E": "Smart, wrong network"}
N_SCREEN = B.N_SCREEN
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
    say(f"  Catalyst fixed at {WCAT:g} g ({A_SPEC:.0f} m2/g); knobs: P, flows of A, B and inert, T.")
    rows = []
    knobs = [4, 0, 3]                       # T, pressure, inert
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
            P, VA, VB, VI, T = ph.T
            F, F0 = S.simulate(P, VA, VB, VI, WCAT, T, truth=True, method="lsoda")
            rel = 100 * S.profit(F, F0, P, WCAT) / best
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
    axes[0].set_ylabel("profit, % of the best possible", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", title="reaction network",
                   title_fontsize=8)
    fig.suptitle(f"Operating window at {WCAT:g} g of catalyst: profit when one knob moves away from the optimum "
                 "(others at optimum). Bars = range keeping >= 90 % of the best profit.", fontsize=9.5, color=INK)
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
    banner("PART 2 - Head to head: smart advisor vs a chemist's rules vs plain BO vs random walk")
    say(f"  {seeds} campaigns per method and chemistry (10 different starting points), {budget} runs each.")
    say(f"  Every method sees the same virtual reactor, the same GC noise and the SAME {N_SCREEN} screening\n"
        "  runs; from then on each proposes runs within the step limits.  Scored by the TRUE profit\n"
        "  of the settings each method would recommend after each run.\n")
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
            if m in ("smart", "chemist"):
                q1, q3 = np.percentile(rel, [25, 75], 0)
                ax.fill_between(n, np.maximum(q1, -60), q3, color=COL[m], alpha=0.15, lw=0)
            ax.plot(n, np.maximum(med, -60), STYLE[m], color=COL[m], lw=2, label=B.LABELS[m])
            ends.append([float(np.clip(med[-1], -55, 100)), SHORT[m]])
        ends.sort(reverse=True)                       # keep end labels >= 7 points apart,
        ends[0][0] = min(ends[0][0], 94)              # below the 100 % line
        for k in range(1, len(ends)):
            ends[k][0] = min(ends[k][0], ends[k - 1][0] - 7)
        for y, txt in ends:
            ax.text(budget + 0.6, y, txt, fontsize=7.5, color=INK2, va="center")
        ax.axhline(100, color=INK2, lw=0.8, ls=":")
        ax.axvspan(0.5, N_SCREEN + 0.5, color=GRID, alpha=0.5, lw=0)
        ax.text(N_SCREEN / 2 + 0.5, -58, f"{N_SCREEN} screening\nruns, same\nfor all", fontsize=6.5, color=INK2,
                ha="center", va="bottom")
        ax.text(N_SCREEN + 0.8, 103, "method chooses, one allowed step per run ->", fontsize=7, color=INK2)
        style(ax)
        ax.set_title(B.SYSTEM_LABEL[system], fontsize=10, color=INK)
        ax.set_xlabel("runs on the reactor", fontsize=9, color=INK)
        ax.set_xlim(0.5, budget + 8)
        ax.set_ylim(-62, 110)
    axes[0].set_ylabel("profit of the recommended settings,\n% of the best possible (median over campaigns)",
                       fontsize=9, color=INK)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Shaded = middle half of the campaigns (smart advisor and chemist). Values below -60 % are drawn at -60 %.",
                 fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig("figures/smart_2_progress.png", dpi=150)
    plt.close(fig)

    # table + stats
    lines = []
    for system in ("simple", "complex"):
        f = true_optimum(system)["f_opt"]
        say(f"\n  {B.SYSTEM_LABEL[system]}   (best possible profit {f:.1f} /h)")
        say(f"    {'method':58s} {'after 10':>9s} {'after 15':>9s} {'after ' + str(budget):>9s} {'runs to 90%':>12s}")
        for m in methods + (["smart_missing_E"] if system == "complex" else []):
            if (system, m) not in res:
                continue
            rel = res[(system, m)]
            r90 = runs_to(rel)
            med90 = np.median(r90)
            hit = np.mean(np.isfinite(r90))
            s90 = f"{med90:g}" if np.isfinite(med90) else f">{budget}"
            s90 += f" ({100 * hit:.0f}% hit)"
            a10, a15, aN = np.median(rel[:, 9]), np.median(rel[:, 14]), np.median(rel[:, -1])
            say(f"    {B.LABELS[m]:58s} {a10:8.0f}% {a15:8.0f}% {aN:8.0f}% {s90:>16s}")
            lines.append((system, m, a10, a15, aN, med90, hit))
        a = res[(system, "smart")]
        for other in ("chemist", "plain_bo"):
            b_ = res[(system, other)]
            k = min(len(a), len(b_))
            for at in (9, 14, budget - 1):
                try:
                    p = wilcoxon(a[:k, at], b_[:k, at]).pvalue
                except ValueError:
                    p = 1.0
                say(f"    smart vs {SHORT[other]:16s} after {at + 1:2d} runs: smart better in "
                    f"{np.sum(a[:k, at] > b_[:k, at])}/{k} paired campaigns, Wilcoxon p = {p:.3g}")
    plot_runs_needed(res, budget)
    say("\n  -> figures/smart_2_progress.png, figures/smart_3_runs_needed.png")
    return res, lines


def plot_runs_needed(res, budget):
    methods = ["smart", "chemist", "plain_bo", "random_walk"]
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
    ax.axhline(N_SCREEN, color=INK2, lw=0.8, ls=":")
    ax.text(1.62, N_SCREEN + 0.4, "screening runs", fontsize=7, color=INK2, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([B.SYSTEM_LABEL["simple"], B.SYSTEM_LABEL["complex"]], fontsize=9)
    ax.set_ylim(-4, budget + 5)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= budget])
    ax.set_ylabel("runs needed (median)", fontsize=9, color=INK)
    ax.set_title("Runs needed to reach 90 % of the best possible profit, screening runs included\n"
                 "(hatched = not reached within the budget; small % = share of campaigns that got there)",
                 fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.tight_layout()
    fig.savefig("figures/smart_3_runs_needed.png", dpi=150)
    plt.close(fig)


# =================================================================================== PART 3
def part3(budget_live=15, seed=0):
    banner("PART 3 - Live: the advisor runs the complex reactor and explains itself")
    S = SYSTEMS["complex"]
    o = true_optimum("complex")
    f = o["f_opt"]
    say(f"  {N_SCREEN} screening runs by the operator, then the advisor chooses, one allowed step per run.\n"
        f"  (best possible profit on this reactor: {f:.1f} /h at " +
        ", ".join(f"{n.split(' ')[0]}={v:.3g}" for n, v in zip(NAMES, o["x_opt_phys"])) +
        " - the advisor does not know this)\n")
    say("  run   P(bar) V_A  V_B  V_I   T(K)   expected      measured  true   model's best is   why")
    t0 = time.time()

    def cb(n, x, r, info):
        P, VA, VB, VI, T = unscale(x)[0]
        if info.get("mode") == "walk":
            why = "stepping toward the predicted best"
        elif info["pred"] - info["best_pred"] > 1.64 * info["sd"]:
            why = "model sure it is better"
        else:
            why = "model unsure here: checking"
        say(f"  {n:3d}  {P:6.0f} {VA:4.0f} {VB:4.0f} {VI:4.0f} {T:6.0f}   {info['pred']:5.1f}±{info['sd']:4.1f}"
            f"   {r['profit']:8.1f} {r['profit_true']:6.1f}   {info['steps_away']:2d} run(s) away    {why}")

    h = run_smart("complex", budget=budget_live, seed=seed, callback=cb)
    adv = h["advisor"]
    say(f"\n  ({time.time() - t0:.0f} s)  Recommended settings now make {h['rec'][-1]:.1f} /h "
        f"= {100 * h['rec'][-1] / f:.0f} % of the best possible, after {budget_live} runs.")

    th_fit, th_true = adv.kin.theta, S.theta_true
    k_f, E_f = theta_to_physical(th_fit)
    k_t, E_t = theta_to_physical(th_true)
    sd = adv.kin.param_sd()[1::2] * R * T_REF / 1000
    names = ["A+B<=>C (main)", "C+B->E (over-reaction)", "A->D (cracking)"]
    say("\n  What it learned (from GC readings only):")
    for j in range(3):
        say(f"    {names[j]:24s} Ea fitted {E_f[j]:5.0f} ± {sd[j]:3.0f} kJ/mol   true {E_t[j]:4.0f}"
            f"    k_ref fitted/true = {k_f[j] / k_t[j]:.2f}")
    Tr = unscale(adv.X)[:, 4]
    order_ok = bool(E_f[1] > E_f[0] and E_f[2] > E_f[0])
    bias = np.median(E_f - E_t)
    say("    (The fit absorbs the product inhibition that the model leaves out, so the absolute numbers are")
    say(f"     biased {'high' if bias > 0 else 'low'} by about {abs(bias):.0f} kJ/mol, and the {len(Tr)} runs cover only "
        f"{Tr.min():.0f}-{Tr.max():.0f} K, which limits how")
    say("     well any activation energy can be pinned down. What decides the best temperature is that the")
    say(f"     unwanted reactions have the HIGHER activation energies, which the model got "
        f"{'right' if order_ok else 'wrong'}; the GP correction")
    say(f"     covers the rest, and the recommended settings are within {100 - 100 * h['rec'][-1] / f:.0f} % "
        f"of the best possible.)")

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
    a1.set_ylabel("log10 rate constant k(T), per m2 of catalyst", fontsize=9, color=INK)
    a1.set_title("Arrhenius plot: learned (dashed, 90 % band) vs true (solid)", fontsize=9.5, color=INK)
    a1.legend(frameon=False, fontsize=7, loc="lower left")
    pred = adv.kin.predict_gc(physical(adv.X, adv.W))
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
    get = {(s, m): (a10, a15, aN, r90, hit) for s, m, a10, a15, aN, r90, hit in lines}
    out = ["# Smart reactor advisor - demo summary", "",
           f"{seeds} campaigns per method and chemistry, {budget} runs each ({N_SCREEN} shared screening runs, "
           f"then one step per run within the step limits; catalyst fixed at {WCAT:g} g). Placeholder kinetics and prices.",
           "Profit = true profit of the settings each method recommends, as % of the best possible.", "",
           "| Chemistry | Method | after 10 runs | after 15 | after " + str(budget) + " | runs to reach 90 % |",
           "|---|---|---|---|---|---|"]
    for (s, m), (a10, a15, aN, r90, hit) in get.items():
        r = f"{r90:g}" if np.isfinite(r90) else f">{budget}"
        out.append(f"| {s} | {B.LABELS[m]} | {a10:.0f} % | {a15:.0f} % | {aN:.0f} % | {r} ({100 * hit:.0f} % of campaigns got there) |")
    out += ["", "## Operating window (range of each knob that keeps >= 90 % of the best profit)", "",
            "| Chemistry | Knob | from | to | share of range |", "|---|---|---|---|---|"]
    for name, knob, lo, hi, frac in windows:
        out.append(f"| {name} | {knob} | {lo:.3g} | {hi:.3g} | {100 * frac:.0f} % |")

    fmt = lambda v: f"{v:g}" if np.isfinite(v) else f"more than {budget}"
    for system in ("simple", "complex"):
        f = true_optimum(system)["f_opt"]
        say(f"\n  {system.upper()} reaction ({N_SCREEN} screening runs + chosen runs):")
        say(f"    {'':18s}{'after 10 runs':>15s}{'after 15 runs':>15s}{'after ' + str(budget):>12s}{'runs to 90 %':>16s}")
        for m, label in (("smart", "smart advisor"), ("chemist", "chemist's rules"), ("plain_bo", "plain BO")):
            a10, a15, aN, r90, hit = get[(system, m)]
            say(f"    {label:18s}{a10:14.0f}%{a15:14.0f}%{aN:11.0f}%{fmt(r90):>16s}")
    gap = {s: get[(s, "smart")][1] - get[(s, "chemist")][1] for s in ("simple", "complex")}
    rs = {s: (get[(s, "smart")][3], get[(s, "chemist")][3]) for s in ("simple", "complex")}
    reading = (
        f"\n  Reading: a good chemist gets there too, given enough runs - after {budget} runs the rules of\n"
        f"  thumb reach {get[('simple', 'chemist')][2]:.0f} % (simple) and {get[('complex', 'chemist')][2]:.0f} % (complex). "
        f"The advisor's value is SPEED:\n  it needs {fmt(rs['simple'][0])} runs (simple) and {fmt(rs['complex'][0])} runs (complex) "
        f"to reach 90 % of the best profit, against\n  {fmt(rs['simple'][1])} and {fmt(rs['complex'][1])} for the chemist, "
        f"and after 15 runs it is {gap['simple']:.0f} points ahead on the simple\n  reaction and {gap['complex']:.0f} points "
        "ahead on the complex one. Each run saved is a day of reactor time, feed\n  gas and operator time. "
        "The advisor is faster because it reads the full GC composition, learns each\n  reaction's temperature "
        "response and the equilibrium limit from the screening runs alone, and then\n  walks straight to the "
        "predicted best, one allowed step at a time, instead of probing one knob\n  at a time. Plain BO, which sees "
        "only profit, cannot plan beyond one step and falls far behind.")
    say(reading)
    out += ["", "## Reading", "", " ".join(reading.split())]
    with open("results/smart_demo_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    say("\n  -> results/smart_demo_summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=25)
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
