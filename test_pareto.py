"""Checks for the profit versus selectivity trade-off curve (smart/pareto.py).
Run:  python test_pareto.py   (~2 to 3 min on a quiet machine, 7 min was measured next to other
heavy jobs; the very first run also computes the true curve of the virtual reactor, about 20 s
more, and keeps it in smart/pareto_true.json)

The limits used below were written down as round numbers for "useful to an operator" and are
not fitted to the outcome; every measured value is printed next to its check so that the
real gap can be read, whatever the limit."""
import sys
import copy
import json
import time
import warnings
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
from smart.systems import LOW, HIGH, MIN_TOTAL_FLOW, KEYS, unscale, feasible
from smart.advisor import SmartAdvisor, VirtualReactor, run_smart, start_campaign, sobol_in, true_optimum
from smart.pareto import (tradeoff, true_tradeoff, true_values, compare_with_truth, spec_for_point,
                          pareto_front, Z90, _knee)

TOL_PLAN = 0.02            # most profitable end vs the advisor's own plan(): 2 % (plan is itself a random search)
TOL_FRONT = 0.05           # an independent search may beat the curve by at most 5 % of its top profit
TOL_SHORTFALL_PCT = 5.0    # profitable end: the settings must truly be within 5 % (of the best profit) of the true curve
TOL_PROFIT_ERR_PCT = 10.0  # profitable end: predicted profit within 10 % (of the best profit) of the truth
TOL_SEL = 0.01             # profitable end: predicted selectivity within 1 percentage point of the truth
TARGET_SECONDS = 10.0      # tradeoff() on a fitted advisor


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def info(text):
    print("INFO " + text)


def arrays(res):
    pts = res["points"]
    return (np.array([p["x"] for p in pts]), np.array([p["profit"] for p in pts], float),
            np.array([p["selectivity"] for p in pts], float))


def non_dominated(P, S):
    """No point has at least the profit AND at least the selectivity of another one."""
    n = len(P)
    return not any(i != j and P[j] >= P[i] and S[j] >= S[i] for i in range(n) for j in range(n))


def curve_ok(res):
    _, P, S = arrays(res)
    return bool(np.all(np.diff(S) > 0) and np.all(np.diff(P) < 0) and non_dominated(P, S))


def plain(o):
    """Only what JSON knows: dict with text keys, list, text, bool, int, finite float, None.
    (json.dumps alone would let a NumPy float through, because it is a kind of float.)"""
    if isinstance(o, dict):
        return all(type(k) is str and plain(v) for k, v in o.items())
    if isinstance(o, list):
        return all(plain(v) for v in o)
    return o is None or type(o) in (str, bool, int) or (type(o) is float and bool(np.isfinite(o)))


# 1. the non-dominated filter on a case that can be checked by eye
p_toy = np.array([10.0, 9.0, 9.5, 5.0, 4.0, 10.0, 1.0])
s_toy = np.array([0.50, 0.70, 0.60, 0.90, 0.80, 0.40, 0.95])
check("non-dominated filter: keeps (10, .50) (9.5, .60) (9, .70) (5, .90) (1, .95), sorted by selectivity",
      pareto_front(p_toy, s_toy).tolist() == [0, 2, 1, 3, 6])
check("non-dominated filter: a point with no prediction (NaN profit or selectivity) is never on the curve",
      pareto_front([5.0, np.nan, 3.0], [0.5, 0.99, 0.7]).tolist() == [0, 2]
      and pareto_front([5.0, 4.0, 3.0], [0.5, np.nan, 0.7]).tolist() == [0, 2])

# 2. a short campaign on the complex virtual reactor (5 screening runs + 7 advisor runs, nothing saved)
t0 = time.time()
camp = run_smart("complex", budget=12, seed=0)
adv = camp["advisor"]
info(f"campaign of {adv.n} runs on the complex reactor took {time.time() - t0:.0f} s; "
     f"kinetic model chosen by the data: {camp['kin_model']}")
check("the advisor is fitted on all 12 runs", adv.n == 12)

# 3. the curve: runtime, and the advisor is left exactly as it was
rng_state = json.dumps(adv.rng.bit_generator.state)
spec_before, X_before = dict(adv.spec), adv.X.copy()
t0 = time.time()
res = tradeoff(adv)
seconds = time.time() - t0
pts = res["points"]
X, P, S = arrays(res)
info(f"tradeoff() took {seconds:.1f} s for {res['n_evaluated']} candidate settings and returned {len(pts)} points")
check(f"runtime of tradeoff() under the {TARGET_SECONDS:.0f} s target (measured {seconds:.1f} s)", seconds < TARGET_SECONDS)
check("the advisor is not changed: its random generator, spec and runs are as before",
      json.dumps(adv.rng.bit_generator.state) == rng_state and adv.spec == spec_before and np.array_equal(adv.X, X_before))
check("about n_points points are returned (12 asked for)", 8 <= len(pts) <= 13)
check("the result can be written as JSON (no NaN, no NumPy types)", bool(json.dumps(res, allow_nan=False)) and plain(res))

# 4. it is a trade-off curve
check("points sorted by rising selectivity, profit falls along the curve, mutually non-dominated", curve_ok(res))
mu, sd = adv.predict(X)
check("profit of every point is the advisor's own prediction, with its 90 % range",
      np.allclose(P, mu, atol=1e-3) and np.allclose([p["profit_lo"] for p in pts], mu - Z90 * sd, atol=2e-3)
      and np.allclose([p["profit_hi"] for p in pts], mu + Z90 * sd, atol=2e-3))
_, XA_m, SC_m, _ = adv._physics(X, adv.thetas, kpis=True)
check("selectivity and conversion are those of the best-estimate chemistry, inside their ranges",
      np.allclose(S, SC_m[0], atol=1e-4) and np.allclose([p["conversion"] for p in pts], XA_m[0], atol=1e-4)
      and all(p["selectivity_lo"] <= p["selectivity"] <= p["selectivity_hi"] for p in pts)
      and all(p["conversion_lo"] <= p["conversion"] <= p["conversion_hi"] for p in pts))

# 5. the settings are allowed
phys = unscale(X)
check("all points inside the box, with enough total flow for the GC and at least 2 mL/min of inert",
      bool(np.all(X >= adv.lo - 1e-12) and np.all(X <= adv.hi + 1e-12) and np.all(feasible(phys))
           and np.all(phys[:, 1:4].sum(1) >= MIN_TOTAL_FLOW - 1e-9) and np.all(phys[:, 3] >= 2.0 - 1e-9)))
check("the set points in physical units match the scaled knobs (T also given in degrees C)",
      all(np.allclose([p[k] for k in KEYS], ph, atol=1e-3) and abs(p["T_C"] - (ph[4] - 273.15)) < 1e-2
          for p, ph in zip(pts, phys)))

# 6. the three marked points
mk = res["markers"]
check("markers: most profitable = first point, most selective = last point, balanced one of the points",
      mk["most_profitable"] == 0 and mk["most_selective"] == len(pts) - 1 and 0 <= mk["balanced"] < len(pts)
      and pts[0]["label"] == "most profitable" and pts[mk["balanced"]]["label"] in ("balanced", "most profitable"))
check("the most selective point still makes a profit", pts[-1]["profit"] >= 0.0)
b = pts[mk["balanced"]]
s_toy2 = np.linspace(0.80, 0.98, 13)
u_toy = (s_toy2 - 0.80) / 0.18
check("balanced point: found on a curve with a knee; none on a straight curve, on one bent the other way, "
      "or on a straight one with rounding wiggles",
      0 < _knee(20 * (1 - u_toy ** 3), s_toy2) < 12 and _knee(20 * (1 - u_toy), s_toy2) == 0
      and _knee(20 * (1 - u_toy) ** 2, s_toy2) == 0
      and _knee(np.round(20 * (1 - u_toy) + 0.02 * np.sin(700 * s_toy2), 3), s_toy2) == 0)
cost_before = (pts[0]["profit"] - b["profit"]) / (100 * (b["selectivity"] - pts[0]["selectivity"]))
cost_after = (b["profit"] - pts[-1]["profit"]) / (100 * (pts[-1]["selectivity"] - b["selectivity"]))
check(f"balanced point: a percentage point of selectivity costs less profit before it ({cost_before:.2f} /h) "
      f"than after it ({cost_after:.2f} /h)", 0 < mk["balanced"] < len(pts) - 1 and cost_before < cost_after)
info(f"most profitable {pts[0]['profit']:.2f} /h at {100 * pts[0]['selectivity']:.1f} % selectivity; balanced "
     f"{b['profit']:.2f} /h at {100 * b['selectivity']:.1f} %; most selective {pts[-1]['profit']:.2f} /h at "
     f"{100 * pts[-1]['selectivity']:.1f} %")

# 7. the most profitable end agrees with the advisor's own predicted best
x_plan, mu_plan, sd_plan = adv.plan(np.random.default_rng(0))
info(f"most profitable point of the curve {P[0]:.3f} /h; the advisor's plan() {mu_plan:.3f} /h "
     f"(difference {100 * (P[0] - mu_plan) / mu_plan:+.2f} %)")
check(f"most profitable end agrees with the advisor's plan() within {100 * TOL_PLAN:.0f} %",
      abs(P[0] - mu_plan) <= TOL_PLAN * abs(mu_plan))

# 8. an independent search does not find settings that clearly beat the curve
rng = np.random.default_rng(7)
cand = np.vstack([sobol_in(adv.lo, adv.hi, 2048, rng, adv.free),
                  np.repeat(X, 40, 0) + rng.normal(0, 0.02, (40 * len(X), 5))])
cand = adv._pin(cand)
cand = cand[feasible(unscale(cand))]
mu_c = adv._predict(cand, thetas=adv.thetas[:1])[0]
s_c = adv._physics(cand, adv.thetas[:1], kpis=True)[2][0]
cs, cp = np.array(res["curve"]["selectivity"]), np.array(res["curve"]["profit"])
ok_c = np.isfinite(s_c) & (mu_c > 0) & (s_c <= cs[-1])
excess = mu_c[ok_c] - np.interp(s_c[ok_c], cs, cp, left=cp[0])
info(f"independent check with {len(cand)} other settings: the best of them lies {excess.max():+.3f} /h "
     f"relative to the curve at the same selectivity ({100 * excess.max() / P[0]:+.2f} % of the top profit)")
check(f"no other setting beats the curve by more than {100 * TOL_FRONT:.0f} % of its top profit",
      excess.max() <= TOL_FRONT * P[0])

# 9. distance information: steps from the last run, and whether there is data nearby
step = np.where(adv.step > 0, adv.step, 1.0)
d_near = np.array([np.min(np.max(np.abs(x - adv.X) / step, axis=1)) for x in X])
check("steps_away is the advisor's own count from the last run",
      all(p["steps_away"] == adv.steps_away(adv.X[-1], x) for p, x in zip(pts, X)))
check("a point is flagged as an extrapolation exactly when it is more than 2 allowed steps from every run",
      all(p["extrapolation"] == bool(d > 2.0 + 1e-9) and p["near_data"] != p["extrapolation"] for p, d in zip(pts, d_near))
      and np.allclose([p["steps_from_nearest_run"] for p in pts], d_near, atol=6e-3))
info(f"{sum(p['extrapolation'] for p in pts)} of {len(pts)} points are flagged as extrapolations "
     f"(farthest: {d_near.max():.1f} steps from the nearest run)")

# 10. reference runs and the plain-word explanation
ref = res["reference"]
check("reference: the advisor's best run and the last run, measured and predicted",
      ref["best_run"]["run"] == adv.recommend() + 1 and ref["last_run"]["run"] == adv.n
      and abs(ref["last_run"]["profit_measured"] - adv.y[-1]) < 1e-3
      and abs(ref["best_run"]["selectivity"] - camp["S_C"][adv.recommend()]) < 1e-3)
text = " ".join(res["explanation"])
check("explanation: a few plain sentences naming the selectivity gain and the profit given up",
      3 <= len(res["explanation"]) <= 5 and "selectivity rises from" in text and "profit falls by" in text)
drop_pct = 100 * (pts[0]["profit"] - b["profit"]) / pts[0]["profit"]
short = lambda v: f"{v:.2f}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}")      # as the text writes a cost
check("explanation: its numbers are those of the marked points (selectivities, profit given up, cost per point)",
      f"selectivity rises from {100 * pts[0]['selectivity']:.1f} % to {100 * b['selectivity']:.1f} %" in text
      and f"profit falls by {drop_pct:.{0 if drop_pct >= 10 else 1 if drop_pct >= 1 else 2}f} % (from {pts[0]['profit']:.1f} to {b['profit']:.1f} per hour)" in text
      and f"about {short(cost_after)} per hour for every further percentage point of selectivity, against {short(cost_before)} up to it" in text
      and f"reach {100 * pts[-1]['selectivity']:.1f} % at {pts[-1]['profit']:.1f} per hour" in text)
for line in res["explanation"]:
    info("  " + line)

# 11. same seed, same curve; another seed, a curve close to it
small = dict(n_candidates=256, rounds=2)
r1, r2 = tradeoff(adv, seed=3, **small), tradeoff(adv, seed=3, **small)
check("deterministic for a given seed", [p["x"] for p in r1["points"]] == [p["x"] for p in r2["points"]])
res_b = tradeoff(adv, seed=1)
gap_seed = np.array(res_b["curve"]["profit"]) - np.interp(res_b["curve"]["selectivity"], cs, cp, left=np.nan, right=np.nan)
info(f"another search seed moves the curve by {np.nanmean(np.abs(gap_seed)):.3f} /h on average "
     f"(largest {np.nanmax(np.abs(gap_seed)):.3f} /h); top profit {res_b['points'][0]['profit']:.3f} vs {P[0]:.3f}")
check("another seed gives the same most profitable end within 1 %", abs(res_b["points"][0]["profit"] - P[0]) <= 0.01 * P[0])

# 12. operator limits: a narrower safety range and a fixed knob (same fitted chemistry, other limits)
adv_box = copy.copy(adv)
adv_box.lo, adv_box.hi = adv.lo.copy(), adv.hi.copy()
adv_box.hi[0] = (40.0 - LOW[0]) / (HIGH[0] - LOW[0])                 # at most 40 bar
adv_box.lo[4] = (540.0 - LOW[4]) / (HIGH[4] - LOW[4])                # never below 540 K
adv_box.fixed = {3: (10.0 - LOW[3]) / (HIGH[3] - LOW[3])}            # inert fixed at 10 mL/min
adv_box.free = [0, 1, 2, 4]
res_box = tradeoff(adv_box, rounds=4)
ph_box = unscale(arrays(res_box)[0])
check("safety range and fixed knob respected (P <= 40 bar, T >= 540 K, inert = 10 mL/min)",
      bool(np.all(ph_box[:, 0] <= 40 + 1e-6) and np.all(ph_box[:, 4] >= 540 - 1e-6) and np.allclose(ph_box[:, 3], 10.0)))
check("the limited curve is a trade-off curve too, and cannot beat the unlimited one at its top",
      curve_ok(res_box) and res_box["points"][0]["profit"] <= P[0] + TOL_PLAN * P[0])
info(f"with those limits: {len(res_box['points'])} points, top profit {res_box['points'][0]['profit']:.2f} /h, "
     f"selectivity {100 * res_box['points'][0]['selectivity']:.1f} to {100 * res_box['points'][-1]['selectivity']:.1f} %")

# 13. flows capped so low that the minimum total flow binds, and nothing makes a profit
adv_low = copy.copy(adv)
adv_low.hi = adv.hi.copy()
adv_low.hi[1] = adv_low.hi[2] = (5.0 - LOW[1]) / (HIGH[1] - LOW[1])  # A and B at most 5 mL/min each
adv_low.hi[3] = (4.0 - LOW[3]) / (HIGH[3] - LOW[3])                  # inert at most 4 mL/min
res_low = tradeoff(adv_low, rounds=3)
ph_low = unscale(arrays(res_low)[0])
check("minimum total flow respected when the flow limits make it bind",
      bool(np.all(ph_low[:, 1:4].sum(1) >= MIN_TOTAL_FLOW - 1e-9) and np.all(ph_low[:, 1:3] <= 5 + 1e-6)))
check("when no allowed setting makes a profit the curve shows the least-loss compromises and says so",
      (res_low["points"][0]["profit"] > 0) or ("None of the allowed settings is predicted to make a profit" in " ".join(res_low["explanation"])))
check("that curve is sorted, non-dominated and JSON too", curve_ok(res_low) and bool(json.dumps(res_low, allow_nan=False)))
info(f"flows capped at 5 / 5 / 4 mL/min: top profit {res_low['points'][0]['profit']:.2f} /h")

# 13b. curves that offer no real choice, and a wanted profit that is out of reach
x_top = np.array(pts[0]["x"])
adv_tiny = copy.copy(adv)
adv_tiny.lo, adv_tiny.hi = np.clip(x_top - 0.002, 0, 1), np.clip(x_top + 0.002, 0, 1)      # +- 0.2 % of every range
res_tiny = tradeoff(adv_tiny, rounds=3)
text_tiny = " ".join(res_tiny["explanation"])
check("a box so small that selectivity cannot change by 0.2 point gives one point and says there is no real trade-off",
      len(res_tiny["points"]) == 1 and set(res_tiny["markers"].values()) == {0}
      and "no real trade-off" in text_tiny and "selectivity rises" not in text_tiny and plain(res_tiny))
adv_fix = copy.copy(adv)
adv_fix.fixed, adv_fix.free = {i: float(adv.X[-1][i]) for i in range(5)}, []
res_fix = tradeoff(adv_fix)
check("with every knob fixed the single allowed setting comes back as the only point",
      len(res_fix["points"]) == 1 and np.allclose(res_fix["points"][0]["x"], adv.X[-1])
      and res_fix["points"][0]["label"] == "most profitable" and plain(res_fix))
res_high = tradeoff(adv, min_profit=100.0, rounds=3)
text_high = " ".join(res_high["explanation"])
check("a minimum profit that no setting reaches: the curve is cut at break-even and the text says so "
      "(and does not claim that nothing makes a profit)",
      res_high["points"][0]["profit"] > 0 and res_high["points"][-1]["profit"] >= 0 and curve_ok(res_high)
      and "None of the allowed settings is predicted to earn the 100 per hour asked for" in text_high
      and "predicted to make a profit" not in text_high)
res_18 = tradeoff(adv, min_profit=18.0, rounds=3)
check("a minimum profit that can be reached cuts the curve there, and the text names it",
      res_18["points"][-1]["profit"] >= 18.0 and curve_ok(res_18)
      and "still earns at least 18 per hour" in " ".join(res_18["explanation"]))

# 14. the whole curve on request (min_profit=None), and an advisor without runs
res_all = tradeoff(adv, min_profit=None, rounds=4)
check("min_profit=None keeps the loss-making selective end as well",
      res_all["points"][-1]["selectivity"] > pts[-1]["selectivity"] and res_all["points"][-1]["profit"] < 0 and curve_ok(res_all))
mk_all, pts_all = res_all["markers"], res_all["points"]
check("on the whole curve the 'most selective' mark is still the last point that makes a profit, the balanced "
      "point makes a profit too, and the text says that the curve goes on at a loss",
      pts_all[mk_all["most_selective"]]["profit"] >= 0 and pts_all[mk_all["most_selective"] + 1]["profit"] < 0
      and mk_all["balanced"] <= mk_all["most_selective"] and pts_all[mk_all["balanced"]]["profit"] > 0
      and pts_all[-1]["label"] == "" and "Past break-even the curve goes on" in " ".join(res_all["explanation"]))
try:
    tradeoff(SmartAdvisor("complex"))
    raised = False
except ValueError as e:
    raised = "no runs yet" in str(e)
check("an advisor without runs gives a clear error instead of a curve", raised)

# 15. 'aim for this point' is the advisor's existing minimum-selectivity spec
i_aim = mk["balanced"] if 0 < mk["balanced"] < len(pts) - 1 else len(pts) // 2
aim = pts[i_aim]
spec = spec_for_point(aim, adv)
check("spec_for_point returns a minimum selectivity close to the point's own", set(spec) == {"S_C"} and abs(spec["S_C"] - aim["selectivity"]) < 0.005)
check("other specs are kept", spec_for_point(aim, spec={"dT_ad": 400.0}) == {"dT_ad": 400.0, "S_C": spec["S_C"]})
adv_aim = copy.copy(adv)
adv_aim.spec = spec
check("under that spec the advisor accepts the point itself", bool(adv_aim._ok(np.array(aim["x"])[None])[0]))
check("under that spec the advisor rejects the most profitable (less selective) point", not bool(adv_aim._ok(X[:1])[0]))
x_s, mu_s, _ = adv_aim.plan(np.random.default_rng(0))
info(f"aiming for the point with {aim['profit']:.2f} /h at {100 * aim['selectivity']:.1f} %: minimum selectivity "
     f"{spec['S_C']:.4f}; the advisor's own plan() under that spec finds {mu_s:.2f} /h")
plans = [mu_s] + [adv_aim.plan(np.random.default_rng(k))[1] for k in (1, 2, 3, 4)]
info(f"OPEN ISSUE (smart/advisor.py, not this module): over 5 random seeds plan() under that minimum selectivity finds "
     f"{min(plans):.2f} to {max(plans):.2f} /h, that is {100 * min(plans) / aim['profit']:.0f} to "
     f"{100 * max(plans) / aim['profit']:.0f} % of the curve point it is meant to aim for, although it accepts that point")
check("the explanation does not promise that the advisor will reach the chosen point",
      "may settle for less profit than the point shows" in text)
check(f"the advisor's plan() under that spec does not beat the curve point by more than {100 * TOL_FRONT:.0f} % of the top profit",
      mu_s <= aim["profit"] + TOL_FRONT * P[0])
res_spec = tradeoff(adv_aim, n_candidates=256, rounds=2)
check("with a minimum selectivity set, every point says whether it meets it",
      res_spec["spec"] == spec and all(isinstance(p["meets_current_spec"], bool) for p in res_spec["points"])
      and not res_spec["points"][0]["meets_current_spec"] and res_spec["points"][-1]["meets_current_spec"])
check("meets_current_spec is the advisor's own verdict on each point",
      [p["meets_current_spec"] for p in res_spec["points"]] == adv_aim._ok(arrays(res_spec)[0]).tolist())
adv_dT = copy.copy(adv)
adv_dT.spec = {"dT_ad": 400.0}                                         # another spec: at most 400 K adiabatic rise
res_dT = tradeoff(adv_dT, rounds=4)
check("with another spec set (maximum adiabatic rise) every returned point is one the advisor's own rule accepts "
      "and carries a minimum selectivity to aim for it",
      bool(np.all(adv_dT._ok(arrays(res_dT)[0]))) and all(p["min_selectivity_spec"] is not None for p in res_dT["points"])
      and curve_ok(res_dT) and plain(res_dT))
info(f"with a maximum adiabatic rise of 400 K: {len(res_dT['points'])} points, top profit {res_dT['points'][0]['profit']:.2f} /h "
     f"(on this chemistry every profitable setting exceeds that rise)")

# 16. the true curve of the virtual reactor (noise-free, fresh catalyst; scoring only)
t0 = time.time()
truth = true_tradeoff("complex")
info(f"true_tradeoff took {time.time() - t0:.1f} s here ({truth['seconds']:.0f} s when it was computed: "
     f"{truth['n_evaluated']} settings, {truth['n_exact']} of them with the exact solver)")
f_opt = true_optimum("complex")["f_opt"]
info(f"top of the true curve {truth['f_best']:.4f} /h; true optimum from smart/optima.json {f_opt:.4f} /h")
check("true curve: sorted, non-dominated, JSON", curve_ok(truth) and bool(json.dumps(truth, allow_nan=False)) and plain(truth))
check("true curve: its most profitable end is the known true optimum within 0.5 %", abs(truth["f_best"] - f_opt) <= 0.005 * f_opt)
Xt = arrays(truth)[0]
vr = VirtualReactor("complex", 0)
check("true_values is the virtual reactor without noise",
      max(abs(vr.true_profit(x) - v) for x, v in zip(Xt[::4], true_values("complex", Xt[::4])["profit"])) < 1e-3)
t0 = time.time()
check("the true curve is remembered (second call returns at once)", true_tradeoff("complex") is truth and time.time() - t0 < 0.5)

# 17. predicted curve against the true curve, after 12 runs
cmp = compare_with_truth(res, truth=truth)
try:
    compare_with_truth(dict(res, system="complex_missing_E"))
    named = False
except ValueError as e:
    named = "system='complex'" in str(e) and "complex_missing_E" in str(e)
check("a curve predicted with a network that is not a virtual reactor's (one the advisor was given, or revised) cannot "
      "silently be scored: the caller is told in plain words to name the reactor (system='complex'); the curve records "
      "the advisor's reactions", named and res["network"] == adv.network()
      and compare_with_truth(dict(res, system="complex_missing_E"), system="complex", truth=truth)["all"]["n"] == len(pts))
check("the comparison can be written as JSON", bool(json.dumps(cmp, allow_nan=False)) and plain(cmp))
tv_2g = true_values("complex", X, wcat=2.0)["profit"]                  # the same settings with 2 g of catalyst
check("the comparison uses the catalyst mass the curve was predicted for",
      res["wcat"] == adv.wcat and res["a_spec"] == adv.a_spec
      and np.allclose([c["profit_true"] for c in compare_with_truth(dict(res, wcat=2.0), truth=truth)["points"]], tv_2g, atol=1e-4)
      and not np.allclose([c["profit_true"] for c in cmp["points"]], tv_2g, atol=0.1))
print("     point   predicted            true at those settings     profit error   below the true curve")
for p, c in zip(pts, cmp["points"]):
    print(f"     {p['label'] or '-':16s} {p['profit']:6.2f} /h {100 * p['selectivity']:5.1f} %    {c['profit_true']:6.2f} /h "
          f"{100 * c['selectivity_true']:5.1f} %    {c['profit_error']:+6.2f} /h      "
          + ("beyond its end" if c["shortfall"] is None else f"{c['shortfall']:+5.2f} /h ({c['shortfall_pct']:+.1f} %)"))
end, whole, bp = cmp["profitable_end"], cmp["all"], cmp["best_point"]
inside = sum(p["profit_lo"] <= c["profit_true"] <= p["profit_hi"] for p, c in zip(pts, cmp["points"]))
info(f"profitable end ({end['n']} points predicted to earn at least 80 % of the top): profit error mean "
     f"{end['profit_error_mean_abs']:.3f} /h, largest {end['profit_error_max_abs']:.3f} /h "
     f"({100 * end['profit_error_max_abs'] / cmp['f_best_true']:.1f} % of the true best {cmp['f_best_true']:.2f}); "
     f"selectivity error largest {100 * end['selectivity_error_max_abs']:.2f} percentage points")
info(f"profitable end: the settings fall short of the true curve by {end['shortfall_pct_mean_abs']:.2f} % on average, "
     f"{end['shortfall_pct_max_abs']:.2f} % at most; vertical gap between the two drawn curves "
     f"{end['curve_gap_mean_abs']:.3f} /h on average, {end['curve_gap_max_abs']:.3f} /h at most")
info(f"whole curve ({whole['n']} points): profit error mean {whole['profit_error_mean_abs']:.3f}, largest "
     f"{whole['profit_error_max_abs']:.3f} /h; shortfall mean {whole['shortfall_pct_mean_abs']:.2f} %, largest "
     f"{whole['shortfall_pct_max_abs']:.2f} %; selectivity error largest {100 * whole['selectivity_error_max_abs']:.2f} points; "
     f"{whole['n_beyond_true_curve']} point(s) beyond the end of the true curve")
info(f"most profitable point: predicted {bp['profit_predicted']:.2f} /h, truly {bp['profit_true']:.2f} /h = "
     f"{bp['pct_of_true_best']:.1f} % of the best possible")
info(f"the true profit lies inside the advisor's stated 90 % range at {inside} of {len(pts)} points")
inside_s = sum(p["selectivity_lo"] <= c["selectivity_true"] <= p["selectivity_hi"] for p, c in zip(pts, cmp["points"]))
inside_x = sum(p["conversion_lo"] <= c["conversion_true"] <= p["conversion_hi"] for p, c in zip(pts, cmp["points"]))
info(f"the true selectivity lies inside its stated range at {inside_s} of {len(pts)} points, the true conversion at "
     f"{inside_x} of {len(pts)} (widest range: selectivity +- {50 * max(p['selectivity_hi'] - p['selectivity_lo'] for p in pts):.2f}, "
     f"conversion +- {50 * max(p['conversion_hi'] - p['conversion_lo'] for p in pts):.2f} percentage points)")
check(f"profitable end: settings truly within {TOL_SHORTFALL_PCT:.0f} % of the true curve "
      f"(measured {end['shortfall_pct_max_abs']:.2f} %)", end["shortfall_pct_max_abs"] <= TOL_SHORTFALL_PCT)
check(f"profitable end: predicted profit within {TOL_PROFIT_ERR_PCT:.0f} % of the truth "
      f"(measured {100 * end['profit_error_max_abs'] / cmp['f_best_true']:.1f} %)",
      100 * end["profit_error_max_abs"] / cmp["f_best_true"] <= TOL_PROFIT_ERR_PCT)
check(f"profitable end: predicted selectivity within {100 * TOL_SEL:.0f} percentage point of the truth "
      f"(measured {100 * end['selectivity_error_max_abs']:.2f})", end["selectivity_error_max_abs"] <= TOL_SEL)

# 18. the v6 committee correction is followed as well (same runs, other correction)
adv6 = SmartAdvisor("complex", seed=0, correction="committee", remaining=5)
adv6.update(camp["X"], camp["gc"], camp["y_obs"], hours=camp["hours"])
res6 = tradeoff(adv6)
X6, P6, _ = arrays(res6)
check("v6 (committee): the curve's profit is that advisor's own prediction, and it is a trade-off curve",
      np.allclose(P6, adv6.predict(X6)[0], atol=1e-3) and curve_ok(res6) and bool(json.dumps(res6, allow_nan=False)))
end6 = compare_with_truth(res6, truth=truth)["profitable_end"]
info(f"v6 on the same 12 runs: profitable end profit error largest {end6['profit_error_max_abs']:.3f} /h, "
     f"shortfall at most {end6['shortfall_pct_max_abs']:.2f} %")

# 19. early in a campaign and on the other chemistry: only the 5 screening runs of the simple reactor
S_s, vr_s, h_s, base_s = start_campaign("simple", seed=1)
adv_s = SmartAdvisor("simple", seed=1)
adv_s.update(np.array(h_s["X"]), np.array(h_s["gc"]), np.array(h_s["y_obs"]), hours=np.array(h_s["hours"]))
res_s = tradeoff(adv_s)
check("simple reactor after the 5 screening runs: a valid curve, JSON, all points allowed",
      curve_ok(res_s) and bool(json.dumps(res_s, allow_nan=False)) and bool(np.all(feasible(unscale(arrays(res_s)[0])))))
info(f"simple reactor, 5 runs: {len(res_s['points'])} points, {sum(p['extrapolation'] for p in res_s['points'])} flagged as "
     f"extrapolations; profit {res_s['points'][0]['profit']:.1f} to {res_s['points'][-1]['profit']:.1f} /h, selectivity "
     f"{100 * res_s['points'][0]['selectivity']:.1f} to {100 * res_s['points'][-1]['selectivity']:.1f} %")

print("\nAll trade-off curve checks passed.")
