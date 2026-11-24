"""Preliminary figure: finding the reaction mechanism, and physics vs a neural network at predicting the GC."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8986", "#e6e5e1", "#fcfcfb"
OK, BAD = "#008300", "#e34948"
F = json.load(open("examples/discovery/finder_results.json"))
A = json.load(open("analysis_v7/gc_accuracy.json"))


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(17, 11), facecolor=SURF)
gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.3, width_ratios=[1, 1.15])

# ---------------------------------------------------------------- (a) candidate networks
ax = fig.add_subplot(gs[0, 0])
style(ax)
CAND = [("truth", "true network", "#2a78d6"), ("close", "close: one reaction order wrong", "#1baf7a"),
        ("similar", "similar: other pathway, no equilibrium", "#eda100"),
        ("far", "far: 2 reactions, nothing makes E", "#eb6834")]
FILES = [("rank:complex_exact.csv", "Exact data, 30 runs"), ("rank:complex_noisy.csv", "With GC noise, 30 runs"),
         ("rank:complex_real.csv", "'Real' reactor (site blocking), 30 runs"),
         ("rank:complex_short_campaign.csv", "Short step-limited campaign, 8 runs")]
y, yt = 0, []
for key, lab in FILES[::-1]:
    R = {r["name"]: r for r in F[key]["ranking"]}
    for c, cl, col in CAND[::-1]:
        v = R[c]["x_noise"]
        ax.barh(y, v, height=0.75, color=col)
        rank = R[c]["rank"]
        ax.text(v * 1.12, y, f"{v:.2g}" + ("   ranked 1st" if rank == 1 else ""), va="center", fontsize=7.5, color=INK)
        y += 1
    yt.append((y - 2.5, lab))
    y += 1.3
ax.set_xscale("log")
ax.axvline(1, color=INK2, ls="--", lw=1)
ax.text(0.92, -1.3, "within the GC noise ←", fontsize=7.5, color=INK2, va="center", ha="right")
ax.set_ylim(-2, y - 0.8)
ax.set_yticks([p for p, _ in yt])
ax.set_yticklabels([l.replace(", ", ",\n") for _, l in yt], fontsize=8, color=INK)
ax.set_xlim(0.05, 900)
ax.set_xlabel("typical miss of the GC readings, as a multiple of the GC noise (log scale; lower is better)", fontsize=8, color=INK2)
ax.set_title("A. You type candidate reactions; the finder ranks them", fontsize=10.5, color=INK, loc="left", pad=36)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CAND]
ax.legend(h, [l for _, l, _ in CAND], fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)

# ---------------------------------------------------------------- (b) automatic discovery outcomes
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(SURF)
ax.axis("off")
ax.set_title("B. No candidates given: the finder works out the reactions from the data", fontsize=10.5, color=INK, loc="left")
ext = A.get("complex_noisy.csv", {}).get("extrapolate", {})
TRUE_COMPLEX = {"A + B <=> C", "B + C -> E", "A -> D"}


def eqs(lst):
    return [e.split(".", 1)[1].split("  r")[0].strip() for e in lst]


rows = [
    ("'Mystery' reactions (not built\nin), 30 runs, GC noise", "nothing",
     F["discover:mystery_noisy.csv"], "A + B ⇌ C,  C → D + B,  A + 2B → E",
     "says it cannot tell if A + 2B → E also\nruns backwards; proposes the run that decides"),
    ("Complex, 'real' reactor\n(site blocking not in the rate laws)", "nothing",
     F["discover:complex_real.csv"], "A + B ⇌ C,  C + B → E,  A → D",
     "first adds a fake 4th reaction, then finds\nsite blocking (K_C 0.033 vs true 0.030) and drops it"),
    ("Complex, short step-limited\ncampaign, only 8 runs", "nothing",
     F["discover:complex_short_campaign.csv"], "A + B ⇌ C,  C + B → E,  A → D",
     "first attempt ended at a wrong network that\ndid not fit; it restarted once and found it"),
    ("Complex, 24 coolest runs", "nothing", None, "A + B ⇌ C,  C + B → E,  A → D", "used for test D below"),
    ("Complex, 30 runs", "'far'", F["repair:far"], "", "repaired"),
    ("Complex, 30 runs", "'similar'", F["repair:similar"], "", "repaired"),
    ("Complex, 30 runs", "'close'", F["repair:close"], "", "repaired"),
]
hdr = ["Data", "Start", "Correct", "Time", "Notes"]
xs = [0.0, 0.265, 0.395, 0.475, 0.545]
for x, t in zip(xs, hdr):
    ax.text(x, 0.97, t, fontsize=8, color=INK2, transform=ax.transAxes, fontweight="bold")
yy = 0.9
for data, start, res, truth_txt, note in rows:
    if res is None:
        found = set(eqs(ext.get("found", []))) if ext else set()
        ok = found == TRUE_COMPLEX
        secs = ext.get("seconds", np.nan)
        if not ext:
            note = "not run yet"
    else:
        found = set(eqs(res["best"]["equations"]))
        truth = TRUE_COMPLEX if "mystery" not in data.lower() else {"A + B <=> C", "C -> B + D", "A + 2 B -> E"}
        ok = found == truth
        secs = res["seconds"]
    ax.text(xs[0], yy, data, fontsize=8, color=INK, transform=ax.transAxes, va="top")
    ax.text(xs[1], yy, start, fontsize=8, color=INK, transform=ax.transAxes, va="top")
    ax.text(xs[2], yy, ("✔ yes" if ok else "✘ no"), fontsize=9, color=OK if ok else BAD,
            transform=ax.transAxes, va="top", fontweight="bold")
    ax.text(xs[3], yy, f"{secs:.0f} s" if np.isfinite(secs) else "–", fontsize=8, color=INK, transform=ax.transAxes, va="top")
    ax.text(xs[4], yy, note, fontsize=7.3, color=INK2, transform=ax.transAxes, va="top")
    ax.plot([0, 1], [yy + 0.022, yy + 0.022], color=GRID, lw=0.8, transform=ax.transAxes)
    yy -= 0.125 if "\n" in data or "\n" in note else 0.085
ax.text(0, yy - 0.01, "True networks: 'mystery' = A + B ⇌ C,  C → D + B,  A + 2B → E;   complex = A + B ⇌ C,  C + B → E,  A → D.\n"
        "Correct = the finder's best network has exactly the true reactions.", fontsize=7.5, color=INK,
        transform=ax.transAxes, va="top")
ax.text(0, yy - 0.11, "Search: one reaction added, removed or swapped at a time from all 22 atom-balanced reactions of A–E,\n"
        "scored by BIC (fit to the GC readings, penalised for every extra rate constant). Times on one laptop core.",
        fontsize=7.3, color=INK2, transform=ax.transAxes, va="top")

# ---------------------------------------------------------------- (c) learned constants vs truth
ax = fig.add_subplot(gs[1, 0])
style(ax, "y")
TRUTH = {"A + B <=> C": (2e-6, 65), "C -> B + D": (1e-6, 110), "A + 2 B -> E": (6e-8, 85)}
RC = F["discover:mystery_noisy.csv"]["best"]["rate_constants"]
for i, rc in enumerate(sorted(RC, key=lambda r: TRUTH[r["reaction"]][1])):
    k0, E0 = TRUTH[rc["reaction"]]
    ax.bar(i - 0.18, E0, width=0.34, color="#c3c2b7", label="true value" if i == 0 else None)
    ax.bar(i + 0.18, rc["Ea_kJ"], width=0.34, color="#2a78d6", label="learned from the data" if i == 0 else None)
    ax.errorbar(i + 0.18, rc["Ea_kJ"], yerr=2 * rc["Ea_sd_kJ"], color=INK, lw=1.2, capsize=3)
    ax.text(i, max(E0, rc["Ea_kJ"]) + 8, f"Ea {rc['Ea_kJ']:.1f} vs {E0} kJ/mol\nk off by {100 * (rc['k_ref'] / k0 - 1):+.1f} %",
            ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels([r for r in sorted(TRUTH, key=lambda r: TRUTH[r][1])], fontsize=9, color=INK)
ax.set_ylim(0, 140)
ax.set_ylabel("activation energy (kJ/mol)", fontsize=8, color=INK2)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("C. How accurate the learned kinetics are ('mystery' data, found network)", fontsize=10.5, color=INK, loc="left")
ax.text(0, -0.17, "Error bars: ±2 standard deviations of the fit. k = rate constant at 300 °C.", fontsize=7.5, color=INK2,
        transform=ax.transAxes)

# ---------------------------------------------------------------- (d) GC prediction: mechanism vs neural network
ax = fig.add_subplot(gs[1, 1])
style(ax)
CASES = [("mystery_noisy.csv", "'Mystery' reactions"), ("complex_noisy.csv", "Complex reactions"),
         ("complex_real.csv", "Complex, 'real' reactor")]
TESTS = [("interpolate", "runs left out at random"), ("extrapolate", "hottest 6 runs (trained cooler)")]
y, yt = 0, []
for f, lab in CASES[::-1]:
    if f not in A:
        continue
    for t, tl in TESTS[::-1]:
        for who, wl, col in (("nn", "neural network alone", "#eb6834"), ("mechanism", "reaction mechanism", "#2a78d6")):
            v = A[f][t][who]["x_noise"]
            mp = A[f][t][who]["rmse_mol_pct"]
            ax.barh(y, v, height=0.75, color=col)
            ax.text(v * 1.1, y, f"{v:.2g}× noise ({mp:.2f} mol-%)", va="center", fontsize=7.3, color=INK)
            y += 1
        yt.append((y - 1.5, f"{lab}:\n{tl}"))
        y += 1.0
    y += 0.8
ax.set_xscale("log")
ax.axvline(1, color=INK2, ls="--", lw=1)
ax.set_xlim(0.2, 600)
ax.set_ylim(-0.8, y - 1.3)
ax.set_yticks([p for p, _ in yt])
ax.set_yticklabels([l for _, l in yt], fontsize=7.6, color=INK)
ax.set_xlabel("typical miss of the predicted GC composition of runs NOT used for training,\n"
              "as a multiple of the GC noise (log scale; 1 = as good as the GC itself)", fontsize=8, color=INK2)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2a78d6", "#eb6834")]
ax.legend(h, ["reaction mechanism (rate laws + mole balances)", "neural network alone (knobs → GC)"], fontsize=7.8,
          frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_title("D. Predicting the GC of unseen runs: mechanism vs neural network", fontsize=10.5, color=INK, loc="left", pad=22)

fig.text(0.01, 0.005, "PRELIMINARY: reaction finder before its final review. Simulated data, placeholder kinetics. "
         "Neural network: ensemble of 5 two-layer networks (32 units), penalty chosen by cross-validation on the training runs.\n"
         "Fairness check (runs left out at random): a Gaussian process on the same data misses by 5.9-8.0 mol-%, about the same as the "
         "network (5.6-7.4); always predicting the average misses by 12.7-13.8 mol-%. 30 runs are too few for any black box.",
         fontsize=7.5, color=INK2)
out = "figures/smart_7_prelim_mechanism.png"
fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
print("->", out)
