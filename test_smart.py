"""Checks for the smart advisor (smart/ and reactor_advisor.py).  Run:  python test_smart.py  (~1-2 min)"""
import os
import copy
import tempfile
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from reactor_twin import simulate
from smart.systems import SIMPLE, COMPLEX, V_I, unscale, scale, LOW, HIGH
from smart.kinetics import KineticFit
from smart.advisor import SmartAdvisor, VirtualReactor, profit_from_gc, run_ofat
import reactor_advisor as RA


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


rng = np.random.default_rng(0)
X = rng.random((300, 5))
P, VA, VB, T, W = unscale(X).T

# 1. batch simulator (exact solver, no inhibition) reproduces reactor_twin.py
F, F0 = SIMPLE.simulate(P[:30], VA[:30], VB[:30], V_I, W[:30], T[:30], theta=SIMPLE.theta_true, method="lsoda")
ref = np.array([list(simulate(V_A=a, V_B=b, V_I=V_I, Wcat=w, T=t, P=p)["F_out"].values())
                for p, a, b, t, w in zip(P[:30], VA[:30], VB[:30], T[:30], W[:30])])
check("batch simulator == reactor_twin.simulate (simple network)",
      np.max(np.abs(F - ref) / ref.sum(1, keepdims=True)) < 1e-6)

# 2. fast Rosenbrock integrator agrees with LSODA (both systems, true reactor incl. inhibition)
for S in (SIMPLE, COMPLEX):
    Fl, F0 = S.simulate(P, VA, VB, V_I, W, T, truth=True, method="lsoda")
    Ff, _ = S.simulate(P, VA, VB, V_I, W, T, truth=True)
    check(f"fast integrator within 2e-3 of LSODA ({S.name})",
          np.max(np.abs(Ff - Fl) / F0.sum(1, keepdims=True)) < 2e-3)

# 3. atom balances and inert
Fs, F0s = SIMPLE.simulate(P, VA, VB, V_I, W, T, truth=True)
check("simple: A atoms conserved (A + C + D)", np.allclose(Fs[:, 0] + Fs[:, 2] + Fs[:, 3], F0s[:, 0], rtol=1e-3))
check("simple: B atoms conserved (B + C)", np.allclose(Fs[:, 1] + Fs[:, 2], F0s[:, 1], rtol=1e-3))
Fc, F0c = COMPLEX.simulate(P, VA, VB, V_I, W, T, truth=True, method="lsoda")
check("complex: A atoms conserved (A + C + D + E)",
      np.allclose(Fc[:, 0] + Fc[:, 2] + Fc[:, 3] + Fc[:, 4], F0c[:, 0], rtol=1e-6))
check("complex: B atoms conserved (B + C + 2E)",
      np.allclose(Fc[:, 1] + Fc[:, 2] + 2 * Fc[:, 4], F0c[:, 1], rtol=1e-6))
check("inert passes through unchanged", np.allclose(Fc[:, -1], F0c[:, -1]))

# 4. with only the reversible reaction and lots of catalyst, the outlet sits at equilibrium
S = copy.deepcopy(COMPLEX)
th = S.theta_true.copy()
th[2] = th[4] = -60.0                        # switch off over-reaction and cracking
Tq = np.array([600.0, 650.0, 700.0])
Fq, _ = S.simulate(30.0, 20.0, 20.0, V_I, 5.0, Tq, theta=th, method="lsoda")
y = Fq / Fq.sum(1, keepdims=True)
K = y[:, 2] / (30.0 * y[:, 0] * y[:, 1])
check("reversible reaction reaches K_eq(T) from thermodynamics",
      np.allclose(np.log(K), S.log_keq(Tq)[:, 0], atol=0.02))

# 5. GC -> flows -> profit: exact for a noise-free reading
yc = Fc / Fc.sum(1, keepdims=True)
check("flows reconstructed from GC with the inert as internal standard",
      np.allclose(COMPLEX.flows_from_gc(yc, VA, VB, V_I), Fc, rtol=1e-9))
ph = unscale(X)
check("profit from a noise-free GC reading == profit from true flows",
      np.allclose(profit_from_gc(COMPLEX, ph, yc), COMPLEX.profit(Fc, F0c, P, W), atol=1e-8))

# 6. grouped integration gives smooth finite-difference derivatives (match LSODA's)
kf = KineticFit(SIMPLE)
phx = unscale(rng.random((6, 5)))
th0 = SIMPLE.theta_true
eps = 1e-4
d_fast = (kf._gc(np.tile(phx, (2, 1)), np.repeat(np.vstack([th0 + eps * np.eye(4)[1], th0]), 6, 0),
                 groups=np.tile(np.arange(6), 2)).reshape(2, 6, -1))
d_fast = (d_fast[0] - d_fast[1]) / eps
up = SIMPLE.simulate(*phx[:, :3].T, V_I, phx[:, 4], phx[:, 3], theta=th0 + 1e-3 * np.eye(4)[1], method="lsoda")[0]
dn = SIMPLE.simulate(*phx[:, :3].T, V_I, phx[:, 4], phx[:, 3], theta=th0 - 1e-3 * np.eye(4)[1], method="lsoda")[0]
d_ref = (up / up.sum(1, keepdims=True) - dn / dn.sum(1, keepdims=True)) / 2e-3
check("finite-difference sensitivities with shared steps match the exact solver",
      np.max(np.abs(d_fast - d_ref)) < 0.02 * np.max(np.abs(d_ref)) + 1e-3)

# 7. kinetic fit recovers activation energies from noise-free GC data (model = truth)
S0 = copy.deepcopy(SIMPLE)
S0.K_inhib = 0.0
Xd = rng.random((12, 5))
phd = unscale(Xd)
Fd, _ = S0.simulate(*phd[:, :3].T, V_I, phd[:, 4], phd[:, 3], truth=True, method="lsoda")
kf = KineticFit(S0).fit(phd, Fd / Fd.sum(1, keepdims=True))
Ea_fit = kf.theta[1::2] * 8.314 * 573.15 / 1000
check(f"kinetic fit recovers Ea (fitted {np.round(Ea_fit, 1)} vs true [80, 120] kJ/mol)",
      np.allclose(Ea_fit, [80, 120], atol=3))

# 8. the advisor respects safety limits and fixed knobs
vr = VirtualReactor(SIMPLE, seed=1)
Xs = rng.random((8, 5))
runs = [vr.run(x) for x in Xs]
lo, hi = np.zeros(5), np.ones(5)
lo[3], hi[3] = 0.1, 0.3                        # temperature limited to 490-570 K
adv = SmartAdvisor(SIMPLE, lo=lo, hi=hi, fixed={0: 0.5})
adv.update(Xs, np.array([r["gc"] for r in runs]), np.array([r["profit"] for r in runs]))
x, info = adv.suggest()
check("suggestion stays inside the safety box", 0.1 - 1e-12 <= x[3] <= 0.3 + 1e-12 and (x >= 0).all() and (x <= 1).all())
check("suggestion keeps the fixed knob fixed", abs(x[0] - 0.5) < 1e-12)
check("prediction has a positive uncertainty", info["sd"] > 0 and np.isfinite(info["pred"]))

# 9. operator log: write / read round trip, % auto-detected, °C <-> K
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "log.csv")
    user = np.array([[10, 20, 20, 327, 1.0], [30, 40, 35, 250, 2.0]])
    g = np.array([r["gc"] for r in runs[:2]])
    RA.write_log(path, SIMPLE, user, g)
    u2, g2 = RA.read_log(path, SIMPLE)
    check("log round trip", np.allclose(u2, user) and np.allclose(g2, g, atol=1e-4))
    RA.write_log(path, SIMPLE, user, 100 * g)
    check("GC given in % is recognised", np.allclose(RA.read_log(path, SIMPLE)[1], g, atol=1e-4))
check("temperature conversion °C <-> K", np.allclose(RA.to_user(RA.to_internal(user)), user))

# 10. trial-and-error baseline runs exactly the budget
h = run_ofat("simple", budget=17, seed=0)
check("one-factor-at-a-time uses exactly the budget", len(h["rec"]) == 17 and len(h["X"]) == 17)

print("\nAll smart-advisor checks passed.")
