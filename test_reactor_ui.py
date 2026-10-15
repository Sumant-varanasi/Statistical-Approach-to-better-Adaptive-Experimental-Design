"""Checks for the web interface's campaign logic (reactor_ui.Campaign), without HTTP.
Run:  python test_reactor_ui.py   (~1 min)"""
import io
import sys
import csv
import json
import warnings
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
import reactor_ui as UI
from smart.systems import MAX_STEP


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def expect_error(name, fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        print(f"PASS {name} -> {e}")
        return
    raise AssertionError(name + ": no error raised")


C = UI.Campaign()
st = C.state()
check("fresh state: no runs, complex chemistry, 6 species", st["n"] == 0 and st["settings"]["system"] == "complex" and len(st["meta"]["species"]) == 6)
json.dumps(st, allow_nan=False)

# validation of typed runs
expect_error("rejects too little feed", C.add_run, dict(P=10, VA=1, VB=1, VI=2, T=300, gc=[1, 1, 1, 1, 1, 1]))
expect_error("rejects temperature outside the box", C.add_run, dict(P=10, VA=20, VB=20, VI=10, T=700, gc=[1, 1, 1, 1, 1, 1]))
expect_error("rejects wrong number of GC columns", C.add_run, dict(P=10, VA=20, VB=20, VI=10, T=300, gc=[1, 1, 1]))
expect_error("rejects a GC reading without inert", C.add_run, dict(P=10, VA=20, VB=20, VI=10, T=300, gc=[1, 1, 1, 1, 1, 0]))
expect_error("refuses advice with too few runs", C.analyse)

# example screening runs: within step limits, truth known
st = C.example(dict(seed=0))
rows = np.array([[r["P"], r["VA"], r["VB"], r["VI"], r["T"]] for r in st["runs"]])
check("5 screening runs loaded, consecutive runs within the step limits",
      st["n"] == 5 and st["truth_known"] and bool(np.all(np.abs(np.diff(rows, axis=0)) <= MAX_STEP + 1e-9)))
check("temperatures are reported in °C", 0 < rows[0, 4] < 600)

# GC in % and in fractions give the same run; catalyst mass per run accepted
pct = dict(P=20, VA=20, VB=30, VI=10, T=352, W=2.0, gc=[10, 10, 60, 5, 5, 10])
frac = dict(pct, gc=[0.1, 0.1, 0.6, 0.05, 0.05, 0.1])
r1 = C._parse_run(pct)
r2 = C._parse_run(frac)
check("GC in % == GC in fractions; per-run catalyst mass kept", np.allclose(r1["gc"], r2["gc"]) and r1["W"] == 2.0)
st = C.add_run(pct)
check("a typed run makes the truth unknown (no 'true' lines on a real reactor)", not st["truth_known"] and st["runs"][-1]["source"] == "manual")

# analysis: JSON-safe, suggestion within one step of the last run, inside the box
out = C.analyse(dict(ref="next"))
json.dumps(out, allow_nan=False)
A, s2 = out["analysis"], out["state"]
last = s2["runs"][-1]
nx = A["next"]
d = [abs(nx[k] - last[k]) for k in ("P", "VA", "VB", "VI", "T")]
check("suggestion within one step of the last run", bool(np.all(np.array(d) <= MAX_STEP + 1e-6)))
check("suggestion inside the box and feasible", nx["VA"] + nx["VB"] + nx["VI"] >= 10 and 1 <= nx["P"] <= 100)
check("analysis blocks present", all(k in A for k in ("plan", "worth", "today", "chemistry", "arrhenius", "parity", "landscape", "series")))
check("no true lines when the truth is unknown", all(r["truth"] is None for r in A["arrhenius"]["reactions"]) and all(r["Ea_true"] is None for r in A["chemistry"]["reactions"]))
check("landscape: 5 knobs, 41 points, reachable range inside the box", len(A["landscape"]["knobs"]) == 5 and all(len(k["x"]) == 41 and k["reach"][0] <= k["reach"][1] for k in A["landscape"]["knobs"]))
check("model estimate stored per run", all(r["pred"] is not None for r in s2["runs"]))

# what-if around another reference reuses the fitted advisor
ls = C.landscape(dict(ref="best"))["landscape"]
check("landscape around the best run", ls["ref"] == "best" and len(ls["knobs"]) == 5)

# export / import round trip
text = C.export_log()
before = [(r["P"], r["VA"], r["VB"], r["VI"], r["T"], r["W"], tuple(r["gc"])) for r in C.state()["runs"]]
st = C.import_csv(dict(text=text, replace=True))
after = [(r["P"], r["VA"], r["VB"], r["VI"], r["T"], r["W"], tuple(r["gc"])) for r in st["runs"]]
check("CSV export -> import round trip keeps every run", len(before) == len(after) and all(np.allclose(a[:6], b[:6], atol=1e-3) and np.allclose(a[6], b[6], atol=1e-4) for a, b in zip(before, after)))
expect_error("next_run.csv needs advice first", C.export_next)
C.analyse()
nxt = list(csv.reader(io.StringIO(C.export_next())))
check("next_run.csv has the suggestion in °C with catalyst mass", len(nxt) == 2 and len(nxt[1]) == 9)

# settings: fixed knob honoured, step limits honoured, chemistry switch guarded
expect_error("cannot switch chemistry while runs of the other one are loaded", C.settings, dict(system="simple"))
C.settings(dict(fixed={"3": 5.0}, max_step=[5, 5, 5, 5, 15]))
A = C.analyse()["analysis"]
last = C.state()["runs"][-1]
check("fixed inert honoured and smaller step limits honoured",
      abs(A["next"]["VI"] - 5.0) < 1e-9 and all(abs(A["next"][k] - last[k]) <= lim + 1e-6 for k, lim in zip(("P", "VA", "VB", "VI", "T"), (5, 5, 5, 5, 15))))
expect_error("fixed value outside the safety range is refused", C.settings, dict(lo=[1, 1, 1, 2, 200], hi=[60, 50, 50, 50, 300], fixed={"4": 350}))

# simulate on the virtual reactor appends the suggested run
C.settings(dict(fixed={}, max_step=[10, 10, 10, 10, 30]))
n0 = C.state()["n"]
st = C.simulate({})
check("simulate adds exactly one virtual run at the suggested settings", st["n"] == n0 + 1 and st["runs"][-1]["source"] == "virtual")

# a run that made no product (cold run) is kept and shown with selectivity "-" instead of breaking the page
C.add_run(dict(P=10, VA=20, VB=20, VI=10, T=250, W=1.0, gc=[40, 40, 0, 0, 0, 20]))
st = C.state()
json.dumps(st, allow_nan=False)
check("cold run (no product) is JSON-safe: selectivity null, profit finite",
      st["runs"][-1]["SC"] is None and st["runs"][-1]["profit"] is not None)
json.dumps(UI.finite(dict(a=float("nan"), b=[1.0, float("inf")], c=dict(d=2.0))), allow_nan=False)
check("finite() scrubs NaN and inf anywhere in a response", UI.finite(dict(a=float("nan")))["a"] is None)
C.delete_run(dict(i=C.state()["n"] - 1))

# settings are all-or-nothing: a rejected value changes nothing
before = C.state()["settings"]
expect_error("catalyst mass out of range is refused", C.settings, dict(wcat=100, a_spec=120, hours=6000))
after = C.state()["settings"]
check("rejected settings leave every setting unchanged", after == before)
expect_error("a fixed knob outside a NEW safety range is refused", C.settings, dict(lo=[1, 1, 1, 2, 200], hi=[60, 50, 50, 50, 300], fixed={"4": 350}))
check("safety ranges unchanged after the refusal", C.state()["settings"]["hi"] == before["hi"])

# simulate follows the catalyst mass of the runs in the table, not the setup value
C.settings(dict(wcat=1.0))
C.clear()
C.example(dict(seed=0))
C.add_run(dict(P=20, VA=20, VB=30, VI=10, T=352, W=2.0, gc=[10, 10, 60, 5, 5, 10]))
C.analyse()
st = C.simulate({})
check("virtual run uses the table's catalyst mass (2 g), not the setup's (1 g)", st["runs"][-1]["W"] == 2.0)

# a change to the runs while the advice is being computed discards that advice (never a stale result)
import threading
C.clear()
C.example(dict(seed=1))
errors, results = [], []
def slow_analyse():
    try:
        results.append(C.analyse())
    except ValueError as e:
        errors.append(str(e))
t = threading.Thread(target=slow_analyse)
t.start()
import time as _t
_t.sleep(0.5)
expect_error("a second fit while one is running is refused with a clear message", C.analyse)
C.add_run(dict(P=20, VA=20, VB=30, VI=10, T=352, W=1.0, gc=[10, 10, 60, 5, 5, 10]))
t.join()
check("the fit that was overtaken by a change is discarded, state stays consistent",
      len(results) == 0 and errors and "changed" in errors[0] and C.state()["analysis_ready"] is False)
json.dumps(C.state(), allow_nan=False)
out = C.analyse()
check("fresh fit after the change covers every run", len(out["analysis"]["pred"]) == C.state()["n"])

# switching chemistry with an empty table works and resets prices
C.clear()
st = C.settings(dict(system="simple"))
check("chemistry switched, 5 species, simple prices", st["settings"]["system"] == "simple" and len(st["meta"]["species"]) == 5)
st = C.example(dict(seed=3))
A = C.analyse()["analysis"]
check("simple chemistry: 2 reactions learned", len(A["chemistry"]["reactions"]) == 2 and len(A["arrhenius"]["reactions"]) == 2)

print("\nAll web-interface checks passed.")
