"""Sensitivity sweeps of the digital twin. Produces figures in ./figures"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reactor_twin import simulate, SPECIES

os.makedirs("figures", exist_ok=True)
base = dict(V_A=20, V_B=20, V_I=10, Wcat=1.0, T=600, P=10)

# 1. Profiles along the bed
r = simulate(**{**base, "Wcat": 3.0}, return_profile=True)
W = np.linspace(0, 3.0, 200)
F = r["profile"].sol(W)
plt.figure(figsize=(6, 4))
for i, s in enumerate(SPECIES):
    plt.plot(W, F[i] * 1e6, label=s)
plt.xlabel("Catalyst mass W (g)"); plt.ylabel("Molar flow (µmol/s)")
plt.title("Species flows along the bed (T=600 K, P=10 bar)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/1_bed_profiles.png", dpi=150)

# 2. Temperature sweep: conversion vs selectivity trade-off
Ts = np.linspace(450, 850, 60)
res = [simulate(**{**base, "T": T}) for T in Ts]
plt.figure(figsize=(6, 4))
plt.plot(Ts, [x["X_A"] for x in res], label="Conversion X_A")
plt.plot(Ts, [x["S_C"] for x in res], label="Selectivity S_C")
plt.plot(Ts, [x["Y_C"] for x in res], label="Yield Y_C", lw=2.5)
plt.xlabel("Temperature (K)"); plt.ylabel("Fraction")
plt.title("Temperature sweep (W=1 g, P=10 bar)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/2_temperature_sweep.png", dpi=150)

# 3. Yield map over T and Wcat
Tg = np.linspace(450, 850, 40); Wg = np.linspace(0.05, 5.0, 40)
Y = np.array([[simulate(**{**base, "T": T, "Wcat": w})["Y_C"] for T in Tg] for w in Wg])
plt.figure(figsize=(6, 4.5))
cs = plt.contourf(Tg, Wg, Y, levels=20, cmap="viridis"); plt.colorbar(cs, label="Yield Y_C")
iw, it = np.unravel_index(Y.argmax(), Y.shape)
plt.plot(Tg[it], Wg[iw], "r*", ms=14, label=f"grid best Y_C={Y.max():.3f}")
plt.xlabel("Temperature (K)"); plt.ylabel("Catalyst mass (g)")
plt.title("Yield map (P=10 bar, feed 20/20/10 mL/min)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/3_yield_map_T_W.png", dpi=150)

# 4. Pressure sweep
Ps = np.linspace(1, 100, 50)
res = [simulate(**{**base, "P": P}) for P in Ps]
plt.figure(figsize=(6, 4))
plt.plot(Ps, [x["X_A"] for x in res], label="X_A")
plt.plot(Ps, [x["S_C"] for x in res], label="S_C")
plt.plot(Ps, [x["Y_C"] for x in res], label="Y_C", lw=2.5)
plt.xlabel("Pressure (bar)"); plt.ylabel("Fraction")
plt.title("Pressure sweep (T=600 K, W=1 g)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/4_pressure_sweep.png", dpi=150)

# 5. Cost of one simulation (matters for the surrogate / BO argument)
t0 = time.perf_counter()
for _ in range(200): simulate(**base)
print(f"Mean time per simulation: {(time.perf_counter()-t0)/200*1e3:.2f} ms")
print(f"Grid best on T x Wcat map: Y_C={Y.max():.3f} at T={Tg[it]:.0f} K, W={Wg[iw]:.2f} g "
      f"(cost: {Y.size} evaluations)")
