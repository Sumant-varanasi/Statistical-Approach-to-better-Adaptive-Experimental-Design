"""
Reaction-finder showcase: theoretical data calculated from KNOWN reactions, handed to the reaction
finder (smart/discovery.py) without the answer, and what it returned, good or bad.

Eight cases, one network of reactions and seven ways to make the job harder (few runs, a noisier
GC, one temperature, a weak reaction, a rate law outside the search's choices, an ageing catalyst,
two routes to the same product).  For every case the finder runs its automatic search exactly as
the web page does (same time limit), and the true network is then fitted to the same data so the
result can be judged: did the search miss a better network, or can the data not tell them apart?

    python make_finder_showcase.py            # make the data, run the finder on every case (8 in parallel,
                                              # about 45 minutes, most of it case 7), write examples/discovery/showcase/
    python make_finder_showcase.py --sheet    # rebuild only the workbook from showcase_results.json
    python make_finder_showcase.py --check    # print the product ranges of every case and stop

Writes examples/discovery/showcase/
    <case>.csv                 the run log the finder was given (upload it on the web page: Reaction finder, Upload)
    <case>_answer.txt          the reactions and numbers the data were calculated from
    showcase_results.json      everything the finder returned (the workbook is built from it)
    finder_showcase.xlsx       the data sheet: cases, rate constants, every run, the search log

All kinetic numbers are placeholders, like the rest of the project.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smart import discovery as D
from smart.systems import theta_from, screening_runs, base_case, MAX_STEP
from smart.kinetics import GC_NOISE_REL, GC_NOISE_ABS
import make_discovery_example as MX

OUT = os.path.join(HERE, "examples", "discovery", "showcase")
MAX_SECONDS = 900.0          # the web page's limit for one search
DESIGN_SEED = 21             # a run design not used to develop the search (the worked example uses 7)
WCAT = 1.0

# ------------------------------------------------------------------------------------- the network behind the data
BASE_TEXT = ("A + B <=> C\n"
             "C -> D + B\n"
             "C + B -> E\n")
BASE_K = [2.5e-6, 6.0e-7, 3.0e-7]          # rate constants at 300 C per gram of catalyst (100 m2/g)
BASE_EA = [70e3, 110e3, 120e3]             # activation energies, J/mol
AGE_HALF_LIFE_H = 720.0                    # 'ageing': the catalyst loses half its activity over the 30 runs (30 x 24 h)

CASES = [
    dict(key="clean", title="A designed kinetic study",
         tests="The easy case: 30 runs spread over the whole operating window, normal GC noise.",
         text=BASE_TEXT, k=BASE_K, ea=BASE_EA, design="study", noise=1.0),
    dict(key="few_runs", title="Start of an optimisation campaign (8 runs)",
         tests="Only the 5 shared screening runs and 3 step-limited runs near the base case: a small corner of the window.",
         text=BASE_TEXT, k=BASE_K, ea=BASE_EA, design="campaign", noise=1.0),
    dict(key="noisy_gc", title="A GC 5 times noisier than the finder assumes",
         tests="The same 30 runs, but the GC scatter is 5x the noise model the finder uses (1 % of the reading + 0.05 mol-%).",
         text=BASE_TEXT, k=BASE_K, ea=BASE_EA, design="study", noise=5.0),
    dict(key="one_temperature", title="Every run at one temperature (300 C)",
         tests="Pressure and flows varied, temperature never: activation energies cannot be measured.",
         text=BASE_TEXT, k=BASE_K, ea=BASE_EA, design="one_T", noise=1.0),
    dict(key="weak_reaction", title="A weak side reaction",
         tests="C -> D + B runs 300 times slower, so D stays within a few times the GC noise.",
         text=BASE_TEXT, k=[BASE_K[0], BASE_K[1] / 300, BASE_K[2]], ea=BASE_EA, design="study", noise=1.0),
    dict(key="half_order", title="A rate law outside the search's choices (order 0.5)",
         tests="C + B -> E runs at r = k P_C P_B^0.5. The search only tries orders 0, 1 and 2.",
         text="A + B <=> C\nC -> D + B\nC + B -> E ; orders C=1, B=0.5\n", k=BASE_K, ea=BASE_EA, design="study", noise=1.0),
    dict(key="ageing", title="A catalyst that ages during the study",
         tests="The catalyst loses half its activity over the 30 runs (720 h). The finder assumes nothing changes with time.",
         text=BASE_TEXT, k=BASE_K, ea=BASE_EA, design="study", noise=1.0, ageing=True),
    dict(key="two_routes", title="Two routes to the same product",
         tests="D is made two ways at once: straight from A, and from C falling apart. Can the data separate them?",
         text="A + B <=> C\nA -> D\nC -> D + B\n", k=[2.5e-6, 3.0e-7, 6.0e-7], ea=[70e3, 140e3, 110e3],
         design="study", noise=1.0),
]
NOISE_SEED = {c["key"]: 100 + i for i, c in enumerate(CASES)}


# ===================================================================================== data
def design(kind):
    """Rows P bar, V_A, V_B, V_I mL/min, T kelvin, catalyst g."""
    if kind == "campaign":
        X = screening_runs(base_case(0))
        X = np.vstack([X, X[-1] + [5, 0, 0, 0, 20], X[-1] + [10, 5, 0, 0, 40], X[-1] + [10, 10, -5, 0, 60]])
        assert np.all(np.abs(np.diff(X, axis=0)) <= MAX_STEP + 1e-9)
        return np.column_stack([X, np.full(len(X), WCAT)])
    phys = MX.study_design(seed=DESIGN_SEED)
    if kind == "one_T":
        phys[:, 4] = 300.0 + 273.15
    return phys


def true_gc(c, phys):
    """Noise-free GC mole fractions from the case's reactions (accurate solver).  For an ageing catalyst
    every rate constant of run i is multiplied by its activity 0.5 ** (hours on stream at mid-run / half-life)."""
    S = D.build_system(c["text"], theta=theta_from(c["k"], c["ea"]), name=c["key"])
    if not c.get("ageing"):
        return MX.exact_gc(S, phys), np.ones(len(phys))
    act = 0.5 ** ((24.0 * (np.arange(len(phys)) + 0.5)) / AGE_HALF_LIFE_H)
    out = []
    for row, a in zip(phys, act):
        th = np.array(S.theta_true, float).copy()
        th[0::2] += np.log(a)
        P, VA, VB, VI, T, W = row
        F, _ = S.simulate(P, VA, VB, VI, W, T, theta=th, method="lsoda")
        out.append(F[0] / F[0].sum())
    return np.array(out), act


def measured_gc(y, factor, seed):
    """The finder's GC error model (1 % of the reading + 0.05 mol-%, normal), times `factor`; negative
    readings set to zero, rows normalised to 100 %."""
    rng = np.random.default_rng(seed)
    yn = y * (1 + factor * GC_NOISE_REL * rng.standard_normal(y.shape)) + factor * GC_NOISE_ABS * rng.standard_normal(y.shape)
    yn = np.maximum(yn, 0.0)
    return yn / yn.sum(1, keepdims=True)


def case_data(c):
    phys = design(c["design"])
    y_true, act = true_gc(c, phys)
    y = measured_gc(y_true, c["noise"], NOISE_SEED[c["key"]])
    return phys, y_true, y, act


# ===================================================================================== the finder
def eqs(text_or_lines):
    """Set of reactions without their rate-law orders, in canonical form (A + B -> C and B + A -> C are the same)."""
    return {r.key().split(";")[0] for r in D.parse_network(text_or_lines)}


def outcome(truth_text, found_lines):
    if not found_lines:
        return "No reaction found"
    if D.network_key(D.parse_network(truth_text)) == D.network_key(D.parse_network(found_lines)):
        return "Found exactly"
    t, f = eqs(truth_text), eqs(found_lines)
    if t == f:
        return "Right reactions, a rate law differs"
    if t < f:
        return "Right reactions plus an extra one"
    if f < t:
        return "Missed a reaction"
    if len(t - f) == 1 and len(f - t) == 1:
        return "One reaction replaced by another"
    return "A different network"


def slim_fit(f):
    keep = ("name", "reactions", "equations", "verdict", "fits_noise", "x_noise", "bic", "delta_bic", "tied_with_best",
            "model", "model_words", "rate_constants", "inhibition", "miss", "predicted_gc", "n_params", "differs_by", "not_made")
    return {k: f[k] for k in keep if k in f}


def run_case(c):
    t0 = time.time()
    phys, y_true, y, act = case_data(c)
    data = D.make_data(phys, y, name=f"{c['key']}.csv")
    res = D.discover(data, max_seconds=MAX_SECONDS)
    best = res["best"]
    cands = {"true network": c["text"]}
    same = bool(best["reactions"]) and D.network_key(D.parse_network(c["text"])) == D.network_key(D.parse_network(best["reactions"]))
    if best["reactions"] and not same:
        cands["found by the search"] = "\n".join(best["reactions"])
    rk = D.rank_candidates(cands, data)
    by = {f["name"]: f for f in rk["ranking"]}
    return dict(key=c["key"], phys=phys.tolist(), gc_true=y_true.tolist(), gc=y.tolist(), activity=act.tolist(),
                best=slim_fit(best), runners_up=[slim_fit(f) for f in res["runners_up"][:4]],
                summary=res["summary"], path=res["path"], seconds=res["seconds"], restarts=res["restarts"],
                stopped_early=res["stopped_early"], undecided=res["undecided"], detected=res["detected"],
                cautions=res["cautions"], outcome=outcome(c["text"], best["reactions"]), same_as_truth=same,
                truth_fit=slim_fit(by["true network"]),
                found_refit=slim_fit(by.get("found by the search", by["true network"])),
                wall_s=time.time() - t0)


def write_inputs():
    os.makedirs(OUT, exist_ok=True)
    for c in CASES:
        phys, y_true, y, act = case_data(c)
        with open(os.path.join(OUT, f"{c['key']}.csv"), "w", newline="", encoding="utf-8") as fh:
            fh.write(D.data_to_csv(D.make_data(phys, y, name=f"{c['key']}.csv")))
        lines = [f"# {c['title']}: the reactions {c['key']}.csv was calculated from (the answer).",
                 f"# {c['tests']}",
                 "# Rate constants at 300 C per gram of catalyst (100 m2/g) and activation energies:"]
        lines += [f"#   k{j} = {k:g}, Ea{j} = {e / 1000:g} kJ/mol" for j, (k, e) in enumerate(zip(c["k"], c["ea"]), 1)]
        if c.get("ageing"):
            lines.append(f"#   every rate constant falls with time on stream: x 0.5 ** (hours / {AGE_HALF_LIFE_H:g}); "
                         f"activity {act[0]:.2f} in run 1, {act[-1]:.2f} in run {len(act)}")
        lines.append(f"#   GC noise: {c['noise']:g} x (1 % of the reading + 0.05 mol-%)")
        with open(os.path.join(OUT, f"{c['key']}_answer.txt"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n" + c["text"])


def check():
    for c in CASES:
        phys, y_true, y, act = case_data(c)
        rng = ", ".join(f"{s} {100 * y_true[:, i].min():.2f}-{100 * y_true[:, i].max():.1f}" for i, s in enumerate("ABCDE"))
        print(f"{c['key']:16s} {len(phys):2d} runs  mol-%: {rng}")


def run_all():
    from concurrent.futures import ProcessPoolExecutor
    write_inputs()
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=len(CASES)) as ex:
        futs = {ex.submit(run_case, c): c["key"] for c in CASES}
        for fu in futs:
            k = futs[fu]
            results[k] = fu.result()
            r = results[k]
            print(f"[{time.time() - t0:5.0f} s] {k}: {r['outcome']}; found {'; '.join(r['best']['reactions'])} "
                  f"({r['best']['verdict']}, {r['seconds']:.0f} s)", flush=True)
    with open(os.path.join(OUT, "showcase_results.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(made=time.strftime("%Y-%m-%d"), max_seconds=MAX_SECONDS, results=results), fh, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.check:
        check()
    elif a.sheet:
        import finder_showcase_sheet
        finder_showcase_sheet.build()
    else:
        run_all()
        import finder_showcase_sheet
        finder_showcase_sheet.build()
