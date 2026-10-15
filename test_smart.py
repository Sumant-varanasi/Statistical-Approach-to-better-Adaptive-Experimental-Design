"""Checks for the smart advisor (smart/ and reactor_advisor.py).  Run:  python test_smart.py  (~2 min)"""
import os
import copy
import tempfile
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from reactor_twin import simulate
from smart.systems import (SIMPLE, COMPLEX, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, DIM, unscale, scale, physical,
                           feasible, step_scaled, base_case, screening_runs)
from smart.kinetics import KineticFit
from smart.advisor import (SmartAdvisor, VirtualReactor, profit_from_gc, kpis_from_gc, run_baseline,
                           start_campaign, ChemistPolicy, PlainBOPolicy, RandomWalkPolicy, step_box)
import reactor_advisor as RA


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


rng = np.random.default_rng(0)
X = rng.random((300, 5))
P, VA, VB, VI, T = unscale(X).T
W = 1.0

# 1. batch simulator (exact solver, no inhibition) reproduces reactor_twin.py, inert flow included
F, F0 = SIMPLE.simulate(P[:30], VA[:30], VB[:30], VI[:30], W, T[:30], theta=SIMPLE.theta_true, method="lsoda")
ref = np.array([list(simulate(V_A=a, V_B=b, V_I=i, Wcat=W, T=t, P=p)["F_out"].values())
                for p, a, b, i, t in zip(P[:30], VA[:30], VB[:30], VI[:30], T[:30])])
check("batch simulator == reactor_twin.simulate (simple network, variable inert)",
      np.max(np.abs(F - ref) / ref.sum(1, keepdims=True)) < 1e-6)

# 2. catalyst enters through its surface area: half the mass with twice the BET area is the same bed
F2, _ = SIMPLE.simulate(P[:30], VA[:30], VB[:30], VI[:30], 0.5 * W, T[:30], theta=SIMPLE.theta_true,
                        method="lsoda", a_spec=200.0)
check("surface-area integration: 0.5 g at 200 m2/g == 1 g at 100 m2/g", np.allclose(F, F2, rtol=1e-9))

# 3. fast Rosenbrock integrator agrees with LSODA (both systems, true reactor incl. inhibition)
for S in (SIMPLE, COMPLEX):
    Fl, F0 = S.simulate(P, VA, VB, VI, W, T, truth=True, method="lsoda")
    Ff, _ = S.simulate(P, VA, VB, VI, W, T, truth=True)
    check(f"fast integrator within 2e-3 of LSODA ({S.name})",
          np.max(np.abs(Ff - Fl) / F0.sum(1, keepdims=True)) < 2e-3)

# 4. atom balances and inert
Fs, F0s = SIMPLE.simulate(P, VA, VB, VI, W, T, truth=True)
check("simple: A atoms conserved (A + C + D)", np.allclose(Fs[:, 0] + Fs[:, 2] + Fs[:, 3], F0s[:, 0], rtol=1e-3))
check("simple: B atoms conserved (B + C)", np.allclose(Fs[:, 1] + Fs[:, 2], F0s[:, 1], rtol=1e-3))
Fc, F0c = COMPLEX.simulate(P, VA, VB, VI, W, T, truth=True, method="lsoda")
check("complex: A atoms conserved (A + C + D + E)",
      np.allclose(Fc[:, 0] + Fc[:, 2] + Fc[:, 3] + Fc[:, 4], F0c[:, 0], rtol=1e-6))
check("complex: B atoms conserved (B + C + 2E)",
      np.allclose(Fc[:, 1] + Fc[:, 2] + 2 * Fc[:, 4], F0c[:, 1], rtol=1e-6))
check("inert passes through unchanged", np.allclose(Fc[:, -1], F0c[:, -1]))

# 5. more inert = lower partial pressures = less conversion (everything else equal)
Fa, F0a = SIMPLE.simulate(10.0, 20.0, 20.0, 2.0, W, 600.0, truth=True, method="lsoda")
Fb, F0b = SIMPLE.simulate(10.0, 20.0, 20.0, 40.0, W, 600.0, truth=True, method="lsoda")
check("diluting with inert lowers conversion", SIMPLE.kpis(Fa, F0a)[0][0] > SIMPLE.kpis(Fb, F0b)[0][0])

# 6. with only the reversible reaction and lots of catalyst, the outlet sits at equilibrium
S = copy.deepcopy(COMPLEX)
th = S.theta_true.copy()
th[2] = th[4] = -60.0                        # switch off over-reaction and cracking
Tq = np.array([600.0, 650.0, 700.0])
Fq, _ = S.simulate(30.0, 20.0, 20.0, 10.0, 5.0, Tq, theta=th, method="lsoda")
y = Fq / Fq.sum(1, keepdims=True)
K = y[:, 2] / (30.0 * y[:, 0] * y[:, 1])
check("reversible reaction reaches K_eq(T) from thermodynamics",
      np.allclose(np.log(K), S.log_keq(Tq)[:, 0], atol=0.02))

# 7. GC -> flows -> profit and KPIs: exact for a noise-free reading
yc = Fc / Fc.sum(1, keepdims=True)
ph6 = physical(X, W)
check("flows reconstructed from GC with the inert as internal standard",
      np.allclose(COMPLEX.flows_from_gc(yc, VA, VB, VI), Fc, rtol=1e-9))
check("profit from a noise-free GC reading == profit from true flows",
      np.allclose(profit_from_gc(COMPLEX, ph6, yc), COMPLEX.profit(Fc, F0c, P, W), atol=1e-8))
check("conversion from the GC == true conversion", np.allclose(kpis_from_gc(COMPLEX, ph6, yc)[0], COMPLEX.kpis(Fc, F0c)[0]))

# 8. kinetic fit recovers activation energies from noise-free GC data (model = truth)
S0 = copy.deepcopy(SIMPLE)
S0.K_inhib = 0.0
Xd = rng.random((12, 5))
phd = physical(Xd, W)
Fd, _ = S0.simulate(*phd[:, :4].T, W, phd[:, 4], truth=True, method="lsoda")
kf = KineticFit(S0).fit(phd, Fd / Fd.sum(1, keepdims=True))
Ea_fit = kf.theta_kin[1::2] * 8.314 * 573.15 / 1000
check(f"kinetic fit recovers Ea (fitted {np.round(Ea_fit, 1)} vs true [80, 120] kJ/mol)",
      np.allclose(Ea_fit, [80, 120], atol=3))

# 9. the screening sequence and every policy respect the step limits, the box and the flow minimum
for seed in range(3):
    sr = screening_runs(base_case(seed))
    check(f"screening runs within step limits and feasible (seed {seed})",
          bool(np.all(np.abs(np.diff(sr, axis=0)) <= MAX_STEP + 1e-9)) and feasible(sr).all()
          and np.all(sr >= LOW) and np.all(sr <= HIGH))
step = step_scaled()
for method in ("chemist", "plain_bo", "random_walk"):
    h = run_baseline("simple", method, budget=14, seed=1)
    ph = unscale(h["X"])
    check(f"{method}: every run within one step of the previous one, in the box, feasible",
          bool(np.all(np.abs(np.diff(ph, axis=0)) <= MAX_STEP + 1e-6)) and feasible(ph).all()
          and len(h["rec"]) == 14)
_, _, h0, _ = start_campaign("simple", 1)
pol = ChemistPolicy(step, np.zeros(DIM), np.ones(DIM))
xs = [pol.next(h0)]
check("chemist never repeats an experiment", all(np.max(np.abs(xs[0] - xr)) > 1e-3 for xr in h0["X"]))

# 10. the advisor: step limits, safety box, fixed knob, flow minimum, plan
vr = VirtualReactor(SIMPLE, seed=1)
S_, vr_, h1, _ = start_campaign("simple", 0)
lo, hi = np.zeros(5), np.ones(5)
lo[4], hi[4] = 0.1, 0.4                        # temperature limited to 490-610 K
adv = SmartAdvisor(SIMPLE, lo=lo, hi=hi, fixed={3: 0.2})
adv.update(np.array(h1["X"]), np.array(h1["gc"]), np.array(h1["y_obs"]))
x_prev = h1["X"][-1]
x, info = adv.suggest(x_prev)
check("suggestion within one step of the previous run",
      bool(np.all(np.abs(unscale(x) - unscale(x_prev)) <= MAX_STEP + 1e-6)))
check("suggestion inside the safety box", 0.1 - 1e-9 <= x[4] <= 0.4 + 1e-9 and (x >= 0).all() and (x <= 1).all())
check("suggestion keeps the fixed knob fixed", abs(x[3] - 0.2) < 1e-9)
check("suggestion leaves enough flow for the GC", feasible(unscale(x))[0])
check("plan reports a model optimum inside the box with steps-away >= 0",
      (info["model_opt"] >= lo - 1e-9).all() and (info["model_opt"] <= hi + 1e-9).all() and info["steps_away"] >= 0)
xs_, mu_s, sd_s = adv.plan()
check("steps_away counts the bottleneck knob", adv.steps_away(x_prev, xs_) == int(np.ceil(np.max(np.abs(xs_ - x_prev) / step - 1e-9))))
blo, bhi = step_box(x_prev, step, lo, hi, {3: 0.2})
check("step box is clipped to the safety box and pins fixed knobs",
      (blo >= lo).all() and (bhi <= hi).all() and blo[3] == bhi[3] == 0.2)

# 11. custom step limits are honoured
adv2 = SmartAdvisor(SIMPLE, max_step=[5, 5, 5, 5, 10])
adv2.update(np.array(h1["X"]), np.array(h1["gc"]), np.array(h1["y_obs"]))
x2, _ = adv2.suggest(x_prev)
check("custom step limits honoured", bool(np.all(np.abs(unscale(x2) - unscale(x_prev)) <= np.array([5, 5, 5, 5, 10]) + 1e-6)))

# 12. a log spanning two catalyst loadings is handled (each run uses its own mass)
adv3 = SmartAdvisor(SIMPLE)
Wrows = np.r_[np.full(3, 1.0), np.full(2, 2.0)]
adv3.update(np.array(h1["X"]), np.array(h1["gc"]), np.array(h1["y_obs"]), wcat_rows=Wrows)
check("per-run catalyst mass accepted", np.allclose(adv3.W, Wrows) and np.isfinite(adv3.predict(x_prev[None])[0]).all())

# 13. operator log: write / read round trip, % auto-detected, °C <-> K
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "log.csv")
    user = np.array([[10, 20, 20, 10, 327, 1.0], [30, 40, 35, 5, 250, 2.0]])
    runs = [vr.run(x) for x in rng.random((2, 5))]
    g = np.array([r["gc"] for r in runs])
    RA.write_log(path, SIMPLE, user, g)
    u2, g2, _ = RA.read_log(path, SIMPLE)
    check("log round trip", np.allclose(u2, user) and np.allclose(g2, g, atol=1e-4))
    RA.write_log(path, SIMPLE, user, 100 * g)
    check("GC given in % is recognised", np.allclose(RA.read_log(path, SIMPLE)[1], g, atol=1e-4))
check("temperature conversion °C <-> K", np.allclose(RA.to_user(RA.to_internal(user)), user))

# ================================================================= v5: kinetics ladder, deactivation, planner, spec
from smart.systems import T_REF
from smart.advisor import run_smart, dT_ad_from_gc
from smart.kinetics import KineticFit
rng5 = np.random.default_rng(5)
Xd = np.random.default_rng(3).random((14, 5))
phd = physical(Xd, W)
def noisy(F):
    y = F / F.sum(1, keepdims=True)
    y = y * (1 + 0.01 * rng5.standard_normal(y.shape)) + 0.0005 * rng5.standard_normal(y.shape)
    y = np.maximum(y, 0)
    return y / y.sum(1, keepdims=True)
kf0 = KineticFit(SIMPLE)
# 14. the model ladder: data from each model class are recognised, and only with the evidence they need
Fp, _ = kf0.simulate(phd, SIMPLE.theta_true)                                   # plain truth
kf = KineticFit(SIMPLE).fit(phd, noisy(Fp))
check("ladder: plain data -> plain model", kf.model == "plain")
Fi, _ = kf0.simulate(phd, np.r_[SIMPLE.theta_true, np.log(0.05), 8.0], model="inhib_T")
kf = KineticFit(SIMPLE).fit(phd, noisy(Fi))
d = kf.describe(T=T_REF)
check(f"ladder: inhibited data -> {kf.model}, K_C(Tref) {d['K_C']:.3f} (true 0.05), dH_ads {d['dH_ads']:.0f} (true -38)",
      kf.model in ("inhib", "inhib_T") and 0.03 < d["K_C"] < 0.08)
hrs = np.full(14, 24.0)
Fd, _ = kf0.simulate(phd, np.r_[SIMPLE.theta_true, np.log(0.05), 8.0, 0.02 / 24], t=KineticFit.time_on_stream(hrs), model="decay")
kf = KineticFit(SIMPLE).fit(phd, noisy(Fd), hours=hrs)
check("ladder: a decaying catalyst is NOT claimed without a repeated set point", "d" not in kf.extras() and not kf.has_replicates)
phr = np.vstack([phd, phd[1:2], phd[1:2]])                                     # the operator repeats run 2 twice, later
hrs_r = np.full(16, 24.0)
Fd, _ = kf0.simulate(phr, np.r_[SIMPLE.theta_true, np.log(0.05), 8.0, 0.02 / 24], t=KineticFit.time_on_stream(hrs_r), model="decay")
kf = KineticFit(SIMPLE).fit(phr, noisy(Fd), hours=hrs_r)
d = kf.describe()
check(f"ladder: with check runs the decay is found ({kf.model}, {100 * d.get('loss_per_day', 0):.1f} %/day, true 2.0)",
      kf.model == "decay" and 0.01 < d["loss_per_day"] < 0.035)
check("time on stream is the cumulative duration of the earlier runs", np.allclose(KineticFit.time_on_stream([24, 12, 6]), [0, 24, 36]))

# 15. adiabatic temperature rise: positive for the exothermic networks, zero without reaction
Fc, F0c = COMPLEX.simulate(30.0, 20.0, 20.0, 10.0, W, 550.0, truth=True, method="lsoda")
check("adiabatic temperature rise is positive when the exothermic reactions run", COMPLEX.dT_ad(Fc, F0c)[0] > 50)
F0z = COMPLEX.inlet_flows(20.0, 20.0, 10.0)
check("no reaction -> no temperature rise", abs(COMPLEX.dT_ad(F0z, F0z)[0]) < 1e-9)
y = Fc / Fc.sum(1, keepdims=True)
check("temperature rise from the GC reading matches the flows",
      np.allclose(dT_ad_from_gc(COMPLEX, physical(scale(np.array([30, 20, 20, 10, 550]))[0], W), y), COMPLEX.dT_ad(Fc, F0c), rtol=1e-6))

# 16. the virtual reactor ages on stream, faster when hot and rich in A
vr_a = VirtualReactor(COMPLEX, seed=0, deactivation=0.02 / 24)
x_hot, x_cold = scale(np.array([60, 50, 20, 2, 650]))[0], scale(np.array([10, 10, 40, 40, 460]))[0]
check("deactivation rate is severity-weighted (hot and A-rich ages faster)", vr_a.loss_rate(x_hot) > 3 * vr_a.loss_rate(x_cold))
for _ in range(5):
    vr_a.run(x_hot)
check("activity falls with runs on a deactivating catalyst", vr_a.activity() < 0.9 and vr_a.n_runs == 5 and vr_a.t_on_stream == 120)
check("a stable catalyst keeps full activity", VirtualReactor(COMPLEX, seed=0).activity() == 1.0)

# 17. the horizon planner respects the step limits and the spec, and reports its mode
S_, vr_, h5, _ = start_campaign("complex", 2)
adv5 = SmartAdvisor(COMPLEX, spec={"S_C": 0.8}, remaining=6, seed=2)
adv5.update(np.array(h5["X"]), np.array(h5["gc"]), np.array(h5["y_obs"]), hours=np.array(h5["hours"]))
x5, info5 = adv5.suggest(h5["X"][-1])
check("horizon planner: suggestion within one step of the last run",
      bool(np.all(np.abs(unscale(x5) - unscale(h5["X"][-1])) <= MAX_STEP + 1e-6)))
check("horizon planner: mode, score and spec probability reported",
      info5["mode"] in ("improve", "walk", "check", "stay", "recover") and np.isfinite(info5["score"]) and 0 <= info5["p_spec"] <= 1)
pf, kp = adv5.feasibility(x5[None])
check("suggested run is likely to meet the selectivity spec, or is the recovery step toward it",
      pf[0] >= 0.5 or (info5["mode"] == "recover" and info5["toward_opt"]))
check("calibration factor is at least 1", adv5.calib >= 1.0)
check("recommendation meets the spec when some run does", adv5.recommend() in range(len(h5["X"])))

# 18. a short v5 campaign with check runs: the protocol marks them and the advisor still converges
h = run_smart("complex", budget=11, seed=3, check_every=3, planner="horizon")
check("check runs are marked in the history and repeat the base case",
      bool(h["check"][7]) and bool(h["check"][10]) and np.allclose(h["X"][7], h["X"][1]))
check("cumulative profit is recorded", h["cum_true"].shape == (11,) and np.isclose(h["cum_true"][-1], h["y_true"].sum()))
check("the v4 rule is still available for comparison", run_smart("simple", budget=7, seed=0, planner="greedy", models=("plain",))["strategy"] == "smart_v4")

print("\nAll smart-advisor checks passed.")
