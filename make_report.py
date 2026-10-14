"""
Build the results web page (report/results.html) from the benchmark outputs.

Needs:  results/bench_summary.json   (python analyse_benchmark.py)
        results/surrogate_study.csv  (python surrogate_study.py, optional)
        figures/bo_1d_steps.png      (python demo_bo_1d.py, optional)
Every headline number on the page is computed here from those files, so re-running the
pipeline after the kinetics change updates the page without hand editing.

Run:  python make_report.py
"""
import base64
import csv
import json
import os
import numpy as np
from bo.problem import global_optimum, SPEC

SHORT = {"yield": "Yield", "margin": "Margin", "margin_noisy": "Margin + noise",
         "margin_spec": "Margin + spec"}
CAPTIONS = {
    "yield": "Maximise the yield of C. Easy problem: a broad plateau near the optimum, about 3 % of the box is within 5 % of the best.",
    "margin": "Maximise the process margin (product value minus feed, compression and catalyst cost). Hard problem: about 90 % of the box loses money and the best region is small.",
    "margin_noisy": "Same margin problem, but every observation carries Gaussian measurement noise (sd 0.5 money units/h), like a GC reading. BO recommends the point with the best predicted mean, not the luckiest reading.",
    "margin_spec": "Margin problem with a product spec: selectivity to C must be at least 0.97. Only operating points that meet the spec count. Constrained BO models the spec with a second GP.",
}
GROUPS = {
    "main": dict(label="BO vs conventional", methods=["gp_ei", "nn_ei", "bo_v1", "simplex", "lhs", "random"]),
    "acq": dict(label="Acquisition functions", methods=["gp_ei", "gp_ucb", "gp_pi", "gp_ts", "gp_ei_q4"]),
    "surrogate": dict(label="GP vs NN", methods=["gp_ei", "gp_ei_raw", "nn_ei", "nn_ts"]),
    "logw": dict(label="Log-scaled inputs", methods=["gp_ei_logw", "gp_ei_logall", "gp_ei", "bo_v1"]),
    "constrained": dict(label="Constrained BO", methods=["cgp_ei", "gp_ei", "bo_v1", "lhs", "random"]),
}


def row(summary, setting, strat):
    for r in summary:
        if r["setting"] == setting and r["strategy"] == strat:
            return r
    return None


def ratio(a, b):
    return b / max(a, 1e-6)


def test(S, setting, strat):
    for t in S["tests"]:
        if t["setting"] == setting and t["strategy"] == strat:
            return t
    return None


def sig(t):
    if t is None:
        return ""
    if t["p_holm"] < 0.05:
        return f" (significant, p {'< 0.001' if t['p_holm'] < 0.001 else '= %.3f' % t['p_holm']})"
    return f" (not significant, p = {t['p_holm']:.2f})"


def findings(S):
    sm = S["summary"]
    out = []
    opt_m = global_optimum("margin")
    gap_m = opt_m["f_opt"] - opt_m["f_median"]

    g, r = row(sm, "margin", "gp_ei"), row(sm, "margin", "random")
    if g and r:
        out.append(dict(label="Process margin vs random search",
                        num=f"{ratio(g['median_final_regret'], r['median_final_regret']):.0f}×", unit="lower regret",
                        text=f"After 50 experiments GP + EI is a median {g['median_final_regret'] * gap_m:.2f} money "
                             f"units/h short of the best profit ({opt_m['f_opt']:.1f}); random search is "
                             f"{r['median_final_regret'] * gap_m:.1f} short{sig(test(S, 'margin', 'random'))}."))
    gy, v1y = row(sm, "yield", "gp_ei"), row(sm, "yield", "bo_v1")
    v1m = row(sm, "margin", "bo_v1")
    if gy and v1y:
        out.append(dict(label="Upgrade vs our original BO",
                        num=f"{ratio(gy['median_final_regret'], v1y['median_final_regret']):.0f}×", unit="lower regret on yield",
                        text=f"From fitting the GP, the rank output transform and polishing the acquisition maximum"
                             f"{sig(test(S, 'yield', 'bo_v1'))}. "
                             f"On the margin problem the median is {ratio(g['median_final_regret'], v1m['median_final_regret']):.1f}× lower"
                             f"{sig(test(S, 'margin', 'bo_v1'))}, because some runs stick in a second optimum."))
    ry = row(sm, "yield", "random")
    if gy and ry:
        e = gy["median_evals_0.01"]
        out.append(dict(label="Yield, within 1 % of best",
                        num=f"{e:.0f}" if e else "–", unit="experiments (median)",
                        text=f"GP + EI got within 1 % of the optimum in {gy['success_0.01']:.0%} of seeds. "
                             f"Random search managed it in {ry['success_0.01']:.0%} of seeds within 50 experiments."))
    c, gc = row(sm, "margin_spec", "cgp_ei"), row(sm, "margin_spec", "gp_ei")
    if c and gc:
        opt_c = global_optimum("margin", constraint=SPEC)
        gap_c = opt_c["f_opt"] - opt_c["f_median"]
        out.append(dict(label="Selectivity spec >= 0.97",
                        num=f"{ratio(c['median_final_regret'], gc['median_final_regret']):.0f}×", unit="lower regret",
                        text=f"Constrained BO vs plain GP + EI, which ignores the spec: median gap to the best in-spec "
                             f"profit ({opt_c['f_opt']:.1f}) of {c['median_final_regret'] * gap_c:.1f} vs "
                             f"{gc['median_final_regret'] * gap_c:.1f} money units/h{sig(test(S, 'margin_spec', 'cgp_ei'))}."))
    acq = [row(sm, "margin", s) for s in ["gp_ei", "gp_ucb", "gp_pi", "gp_ts"]]
    acq = [a for a in acq if a]
    if acq:
        worst = max(acq, key=lambda a: a["median_final_regret"])
        out.append(dict(label="Acquisition function",
                        num="EI", unit="lowest median in every setting (tied with UCB on yield)",
                        text=f"On the margin problem EI beats {worst['label']}{sig(test(S, 'margin', worst['strategy']))}. "
                             f"Against UCB and Thompson sampling EI has the lower median, but the difference is not significant."))
    gl = row(sm, "margin", "gp_ei_logw")
    if gl and g:
        out.append(dict(label="Log-scaled P and catalyst",
                        num=f"{gl['success_0.01']:.0%}", unit=f"of runs within 1 % (was {g['success_0.01']:.0%})",
                        text=f"Letting the GP see pressure and catalyst mass on a log scale lowers the median margin regret "
                             f"from {g['median_final_regret']:.4f} to {gl['median_final_regret']:.4f}{sig(test(S, 'margin', 'gp_ei_logw'))}."))
    return out


def surrogate_block():
    p = "results/surrogate_study.csv"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for obj in sorted({r["objective"] for r in rows}):
        sizes = sorted({int(r["n_train"]) for r in rows})
        models = {}
        for m in sorted({r["model"] for r in rows}):
            models[m] = {}
            for ts in ["whole box", "near optimum"]:
                d = {}
                for metric in ["nrmse", "coverage", "nlpd"]:
                    d[metric] = [float(np.median([float(r[metric]) for r in rows if r["objective"] == obj
                                                  and r["model"] == m and r["test_set"] == ts and int(r["n_train"]) == n]))
                                 for n in sizes]
                models[m][ts] = d
        out[obj] = dict(sizes=sizes, models=models,
                        repeats=len({r["repeat"] for r in rows}))
    return out


def main():
    with open("results/bench_summary.json") as fh:
        S = json.load(fh)
    seeds = max(r["n_seeds"] for r in S["summary"])
    budget = max(r["budget"] for r in S["summary"])
    n_runs = sum(r["n_seeds"] for r in S["summary"])
    fig = ""
    if os.path.exists("figures/bo_1d_steps.png"):
        fig = "data:image/png;base64," + base64.b64encode(open("figures/bo_1d_steps.png", "rb").read()).decode()
    data = dict(
        lede=(f"{len({r['strategy'] for r in S['summary']})} strategies, {len(S['curves'])} problem settings, {seeds} random seeds each, "
              f"{budget} reactor experiments per run. Every curve compares the operating point a method would "
              f"recommend with the true optimum of the twin."),
        meta=[["Runs", f"{n_runs:,}"], ["Seeds", seeds], ["Budget", f"{budget} experiments"],
              ["Knobs", "P, V_A, V_B, T, W_cat"], ["Twin", "isothermal packed bed, A+B→C, A→D"]],
        findings=findings(S),
        limits=[
            "Kinetics and prices are placeholders. Re-run the pipeline once Prof. Khare provides real values.",
            "The twin is cheap (about 1 ms per run), so the true optimum is known. On the real reactor it is not, and regret cannot be measured directly.",
            "On the margin problem about a quarter of GP + EI runs stick in a second local optimum (almost no catalyst, about 675 K). Log-scaled inputs reduce this but do not remove it.",
            "GP + EI beats the original BO significantly on yield, but on the margin problems the difference is not significant with 20 seeds.",
            "Results cover a 50-experiment budget and 5 knobs, with inert flow fixed at 10 mL/min.",
            "The NN ensemble uses 5 small networks; a larger or better-tuned ensemble could change the GP vs NN result.",
            "Noise is simple Gaussian noise on the objective. Real GC error depends on composition and may drift over time.",
        ],
        labels=S["labels"], curves=S["curves"], summary=S["summary"], tests=S["tests"],
        setting_order=["yield", "margin", "margin_noisy", "margin_spec"],
        settings_short=SHORT, setting_captions=CAPTIONS, groups=GROUPS,
        surrogate=surrogate_block(), fig1d=fig,
    )
    html = open("report/template.html").read().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open("report/results.html", "w") as fh:
        fh.write(html)
    print(f"Wrote report/results.html ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
