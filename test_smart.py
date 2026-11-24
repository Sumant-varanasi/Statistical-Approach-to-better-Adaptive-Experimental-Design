"""Checks for the smart advisor (smart/ and reactor_advisor.py).  Run:  python test_smart.py  (~7 min)"""
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

# ================================================================= v6: committee correction
from smart.committee import ResidualCommittee, stacking_weights, MIN_NN

# 19. stacking picks the member that predicts held-out data, and stays on the simplex
lp = np.column_stack([np.full(20, -5.0), np.full(20, -1.0), np.full(20, -3.0)])
w = stacking_weights(lp)
check("stacking: weights sum to 1 and favour the best held-out predictor", abs(w.sum() - 1) < 1e-9 and w[1] > 0.8)
check("stacking: on a tie the prior decides (simpler member first)",
      stacking_weights(np.zeros((10, 2)), [1.0, 0.25])[0] > 0.7)

rng_c = np.random.default_rng(0)
Xc = rng_c.random((14, 5))
phys_c = rng_c.normal(0, 10, 14)
noise_c = 0.25
r_noise = rng_c.normal(0, np.sqrt(noise_c), 14)
cm = ResidualCommittee(seed=0).fit(Xc, r_noise, phys_c, noise_c)
mu0, _, _ = cm.predict(rng_c.random((50, 5)), rng_c.normal(0, 10, 50))
check("committee: correct physics -> the correction stays within the noise level", float(np.max(np.abs(mu0))) < 2 * np.sqrt(noise_c))
r_smooth = 8 * np.sin(3 * Xc[:, 4]) + 4 * Xc[:, 0] + rng_c.normal(0, np.sqrt(noise_c), 14)
cm2 = ResidualCommittee(seed=0).fit(Xc, r_smooth, phys_c, noise_c)
check("committee: error that depends on the knobs -> the GP carries the weight", max(cm2.weights, key=cm2.weights.get) == "gp")
r_phys = 0.3 * phys_c + rng_c.normal(0, np.sqrt(noise_c), 14)
cm4 = ResidualCommittee(seed=0).fit(Xc, r_phys, phys_c, noise_c)
check("committee: error that follows the physics prediction -> the networks carry the weight", max(cm4.weights, key=cm4.weights.get) == "nn")
mu2, sd2, parts2 = cm2.predict(Xc[:3], phys_c[:3])
check("committee: spread includes the members' disagreement", bool(np.all(sd2 + 1e-9 >= parts2["disagreement"])))
cm3 = ResidualCommittee(seed=0).fit(Xc[:MIN_NN - 1], r_smooth[:MIN_NN - 1], phys_c[:MIN_NN - 1], noise_c)
check("committee: networks only join once there are enough runs", "nn" not in cm3.weights and "nn" in cm2.weights)

# 20. the v6 advisor plans within the step limits and keeps v5 available
adv6 = SmartAdvisor(COMPLEX, seed=2, correction="committee", remaining=6)
adv6.update(np.array(h5["X"]), np.array(h5["gc"]), np.array(h5["y_obs"]), hours=np.array(h5["hours"]))
x6, info6 = adv6.suggest(h5["X"][-1])
check("v6: suggestion within one step of the last run",
      bool(np.all(np.abs(unscale(x6) - unscale(h5["X"][-1])) <= MAX_STEP + 1e-6)))
check("v6: committee fitted and weights reported", adv6.committee is not None and abs(sum(adv6.committee.weights.values()) - 1) < 1e-9)
check("v5 default still uses the single GP", SmartAdvisor(COMPLEX, seed=2).correction == "gp")

# ================================================================= v7: hybrid committee, neural network alone, ablations
from smart.committee import ProfitEnsemble, MEMBERS, ALL_MEMBERS, HYBRID_MEMBERS
from smart.advisor import NNOnlyPolicy, POLICIES
from smart import benchmark as B


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# 21. the stand-alone network (member 'direct') takes over when the physics is useless and stays out when it is right
rng_d = np.random.default_rng(7)
nd = 25
Xd7, Xq7 = rng_d.random((nd, 5)), rng_d.random((200, 5))
f7 = lambda Z: 20 * np.sin(3 * Z[:, 4]) + 10 * Z[:, 0] - 8 * Z[:, 1] * Z[:, 2]     # profit: a smooth function of the knobs
y7 = f7(Xd7) + rng_d.normal(0, 0.5, nd)
phys_bad, phys_bad_q = rng_d.normal(0, 10, nd), rng_d.normal(0, 10, 200)            # a "physics" unrelated to the profit
ch = ResidualCommittee(seed=0, members=HYBRID_MEMBERS).fit(Xd7, y7 - phys_bad, phys_bad, 0.25)
check(f"hybrid: useless physics, 25 runs -> the stand-alone network carries the weight ({ch.describe()})",
      max(ch.weights, key=ch.weights.get) == "direct")
mu_h, _, parts_h = ch.predict(Xq7, phys_bad_q)
e_h, e_p = rmse(phys_bad_q + mu_h - f7(Xq7)), rmse(phys_bad_q - f7(Xq7))
check(f"hybrid: and then predicts unseen runs better than the physics alone (error {e_h:.1f} against {e_p:.1f})", e_h < e_p)
check("hybrid: the direct member's prediction of the profit is the stand-alone network's (the physics cancels)",
      np.allclose(parts_h["member_mu"]["direct"] + phys_bad_q, ch.models["direct"].predict(Xq7)[0]))
failed = ch.predict(Xq7[:4], np.full(4, -1e3), phys_ok=np.array([False, True, False, True]))[2]["member_mu"]["direct"]
check("hybrid: where the physics model gave no answer the stand-alone network does not overrule the penalty",
      failed[0] == failed[2] == 0.0 and failed[1] > 900 and failed[3] > 900)
P7, VA7, VB7, VI7, T7 = unscale(Xd7).T
F7, F07 = COMPLEX.simulate(P7, VA7, VB7, VI7, W, T7, truth=True)
phys_ok = COMPLEX.profit(F7, F07, P7, W)                                             # the physics is right: only GC noise is left
cr = ResidualCommittee(seed=0, members=HYBRID_MEMBERS).fit(Xd7, rng_d.normal(0, 0.5, nd), phys_ok, 0.25)
check(f"hybrid: correct physics -> the stand-alone network gets little weight ({100 * cr.weights['direct']:.0f} %)",
      cr.weights["direct"] < 0.1)
pe = ProfitEnsemble(0).fit(Xd7, y7)
check("stand-alone network: predictions stay within the profits seen plus half their range",
      bool(np.all(pe.predict(rng_d.random((500, 5)))[0] <= y7.max() + 0.5 * (y7.max() - y7.min()) + 1e-9))
      and bool(np.all(pe.predict(rng_d.random((500, 5)))[0] >= y7.min() - 0.5 * (y7.max() - y7.min()) - 1e-9)))
# a held-out run must not help the networks that are scored on it: run 0 is far hotter and far more profitable than
# the rest (profit = 100 x the temperature knob), so networks trained without it may promise at most the best of the
# OTHER runs plus half their range; if run 0 itself set that limit they could "predict" it far better than they can
Xl = np.full((12, 5), 0.5)
Xl[:, 4] = 0.2 * rng_d.random(12)
Xl[0, 4] = 1.0
yl = 100 * Xl[:, 4] + rng_d.normal(0, 0.3, 12)
ml, vl = ProfitEnsemble(0).fit(Xl, yl, 0.09).heldout(Xl, yl, [np.array([i]) for i in range(12)])
top_l = yl[1:].max() + 0.5 * (yl[1:].max() - yl[1:].min())
check(f"stand-alone network: a held-out run does not widen the range it is judged against "
      f"(run 0 measured {yl[0]:.0f}, predicted without it {ml[0]:.1f}, limit from the other runs {top_l:.1f})",
      ml[0] <= top_l + 1e-9 and bool(np.all(vl >= 0.09 - 1e-12)))
# and on the smooth data above (6 folds): raise one run's profit by 100; the networks trained without that run must
# not follow it (all folds' networks share one optimiser run, so a small move in no particular direction is expected)
folds7 = ResidualCommittee.folds(nd, np.random.default_rng(0))
i7 = int(np.argmax(y7))
y7b = y7.copy()
y7b[i7] += 100.0
m7a = ProfitEnsemble(0).fit(Xd7, y7, 0.25).heldout(Xd7, y7, folds7)[0]
m7b = ProfitEnsemble(0).fit(Xd7, y7b, 0.25).heldout(Xd7, y7b, folds7)[0]
check(f"stand-alone network: raising a held-out run's profit by 100 moves its own held-out prediction by less than 1 "
      f"({m7a[i7]:.2f} -> {m7b[i7]:.2f})", abs(m7b[i7] - m7a[i7]) < 1.0)

# 22. any subset of members that contains 'none' works; the v6 default is unchanged
for mem in (("none",), ("none", "gp"), ("none", "nn"), ("none", "direct"), ("none", "gp", "direct"), ALL_MEMBERS):
    c = ResidualCommittee(seed=0, members=mem).fit(Xd7[:14], (y7 - phys_bad)[:14], phys_bad[:14], 0.25)
    mu, sd, _ = c.predict(Xq7[:5], phys_bad_q[:5])
    check(f"committee works with members {' + '.join(mem)}",
          tuple(c.weights) == mem and abs(sum(c.weights.values()) - 1) < 1e-9 and np.isfinite(mu).all()
          and bool(np.all(sd > 0)) and c.weight_row() == [c.weights.get(k, 0.0) for k in ALL_MEMBERS])
try:
    ResidualCommittee(members=("gp", "nn"))
    refused = False
except ValueError:
    refused = True
check("committee refuses a member list without 'none'", refused)
check("v6 default members unchanged (none, gp, nn)", ResidualCommittee(seed=0).members == ("none", "gp", "nn") == MEMBERS
      and tuple(cm2.weights) == MEMBERS)
cm2h = ResidualCommittee(seed=0, members=HYBRID_MEMBERS).fit(Xc, r_smooth, phys_c, noise_c)
m6, m7 = cm2.predict(Xc[:3], phys_c[:3])[2]["member_mu"], cm2h.predict(Xc[:3], phys_c[:3])[2]["member_mu"]
check("adding the direct member leaves what the GP and the correction networks compute unchanged",
      all(np.array_equal(m6[k], m7[k]) for k in ("gp", "nn")))

# 23. the v7 (hybrid) advisor: a short campaign, its suggestion, and the logged weights
h7 = run_smart("complex", budget=10, seed=2, correction="hybrid")
adv7 = h7["advisor"]
ph7 = unscale(h7["X"])
check("v7: campaign is named smart_v7 and every run is within one step of the previous one, feasible",
      h7["strategy"] == "smart_v7" and bool(np.all(np.abs(np.diff(ph7, axis=0)) <= MAX_STEP + 1e-6)) and feasible(ph7).all())
x7, info7 = adv7.suggest(h7["X"][-1])
check("v7: suggestion within one step of the last run, in the box, feasible",
      bool(np.all(np.abs(unscale(x7) - unscale(h7["X"][-1])) <= MAX_STEP + 1e-6)) and (x7 >= 0).all() and (x7 <= 1).all()
      and feasible(unscale(x7))[0])
check("v7: the stand-alone network has joined the committee and the weights sum to 1",
      adv7.members == HYBRID_MEMBERS and "direct" in adv7.committee.weights and abs(sum(adv7.committee.weights.values()) - 1) < 1e-9)
w7 = h7["committee_weights"]
check("v7: logged weights are (updates, 4) in the order none / gp / nn / direct, with the member names",
      w7.shape == (5, 4) and list(h7["weight_columns"]) == list(ALL_MEMBERS) and list(h7["committee_members"]) == list(HYBRID_MEMBERS)
      and np.allclose(w7.sum(1), 1) and h7["committee_weights_final"].shape == (4,) and h7["committee_heldout_rmse"].shape == (5, 4))
check("v7: the networks have weight 0 until there are enough runs, and a weight after",
      bool(np.all(w7[:MIN_NN - 5, 2:] == 0)) and bool(np.all(w7[MIN_NN - 5:, 2:] > 0)))
ha = run_smart("simple", budget=7, seed=0, correction="committee", members=("none", "gp"))
check("ablation (v6 without its networks): logged weights (updates, 4), absent members at 0, members named",
      ha["committee_weights"].shape == (2, 4) and bool(np.all(ha["committee_weights"][:, 2:] == 0))
      and list(ha["committee_members"]) == ["none", "gp"] and ha["strategy"] == "smart_v6")
h6 = run_smart("simple", budget=7, seed=0, correction="committee")
check("v6 with its default members: logged weights stay (updates, 3) and nothing else is added",
      h6["committee_weights"].shape == (2, 3) and "weight_columns" not in h6 and h6["strategy"] == "smart_v6"
      and h6["advisor"].members == MEMBERS)
try:
    SmartAdvisor(COMPLEX, correction="hybird")
    refused = False
except ValueError:
    refused = True
check("an unknown correction name is refused instead of silently falling back to v5", refused)

# 24. the neural network alone as a baseline policy, and the benchmark's list of methods
h = run_baseline("simple", "nn_only", budget=14, seed=1)
ph = unscale(h["X"])
check("nn_only: every run within one step of the previous one, in the box, feasible",
      bool(np.all(np.abs(np.diff(ph, axis=0)) <= MAX_STEP + 1e-6)) and feasible(ph).all() and len(h["rec"]) == 14
      and bool(np.all(ph >= LOW - 1e-9)) and bool(np.all(ph <= HIGH + 1e-9)) and h["strategy"] == "nn_only")
pol = NNOnlyPolicy(step, np.zeros(DIM), np.ones(DIM), seed=1)
check("nn_only: recommends one of the tested runs", pol.recommend(h0) in range(len(h0["X"])))
new = ("nn_only", "smart_v7", "smart_v6_nonn", "smart_v6_nogp", "smart_v7_missing_E", "smart_v6_nonn_missing_E", "smart_v6_nogp_missing_E")
check("benchmark: the v7 methods are listed, labelled, and every listed method can be run",
      all(m in B.LABELS for m in new) and all(m in B.SMART_KWARGS or m in POLICIES for m in B.LABELS)
      and all(m in B.STRESS for m in new[4:]) and all(m in B.METHODS for m in new[:4]))
check("benchmark: stress tests do not apply to the simple reaction, everything else applies everywhere",
      not any(B.applies(m, "simple") for m in B.STRESS) and all(B.applies(m, "complex") for m in B.LABELS)
      and all(B.applies(m, "simple") for m in B.METHODS))
check("benchmark: the v5 / v6 methods are run with the same arguments as before",
      B.SMART_KWARGS["smart"] == {} and B.SMART_KWARGS["smart_v6"] == dict(correction="committee")
      and B.SMART_KWARGS["smart_v4"] == dict(planner="greedy", models=("plain",))
      and B.SMART_KWARGS["smart_missing_E"] == dict(model="complex_missing_E")
      and B.SMART_KWARGS["smart_v6_missing_E"] == dict(model="complex_missing_E", correction="committee"))

print("\nAll smart-advisor checks passed.")
