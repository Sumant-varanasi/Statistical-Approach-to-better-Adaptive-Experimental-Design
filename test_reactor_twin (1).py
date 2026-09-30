"""Validation checks for reactor_twin.py. Run: python test_reactor_twin.py"""
import numpy as np
from reactor_twin import simulate, DEFAULT_THETA, mfc_to_molar

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name

base = dict(V_A=20, V_B=20, V_I=10, Wcat=1.0, T=600, P=10)
r = simulate(**base)
F0, Fo = r["F0"], r["F_out"]

# 1. Atom/species balances: every A ends up as A, C or D; every B as B or C
check("A balance: FA+FC+FD = FA0",
      np.isclose(Fo["A"] + Fo["C"] + Fo["D"], F0["A"], rtol=1e-6))
check("B balance: FB+FC = FB0", np.isclose(Fo["B"] + Fo["C"], F0["B"], rtol=1e-6))
check("Inert unchanged", np.isclose(Fo["I"], F0["I"], rtol=1e-10))

# 2. Mole fractions sum to 1, bounded metrics
check("sum y_out = 1", np.isclose(sum(r["y_out"].values()), 1.0))
check("0 <= X_A <= 1", 0 <= r["X_A"] <= 1)
check("Y_C = X_A * S_C", np.isclose(r["Y_C"], r["X_A"] * r["S_C"]))

# 3. Total moles drop by exactly the moles of C formed (A+B->C loses 1 mol per C)
check("FT_out = FT_0 - FC_out", np.isclose(r["FT_out"], r["FT_0"] - Fo["C"], rtol=1e-6))

# 4. No catalyst -> no reaction
check("Wcat = 0 gives X_A = 0", simulate(**{**base, "Wcat": 0.0})["X_A"] == 0)

# 5. Conversion rises with catalyst mass
Xs = [simulate(**{**base, "Wcat": w})["X_A"] for w in [0.1, 0.5, 1, 2, 4]]
check("X_A increases with Wcat", all(np.diff(Xs) > 0))

# 6. Analytical check: switch off reaction 1. A -> D is equimolar, so FT is constant and
#    FA = FA0 * exp(-k2 * P * W / FT0)
th = {**DEFAULT_THETA, "k1_ref": 0.0}
r2 = simulate(**base, theta=th)
k2 = th["k2_ref"] * np.exp(-th["Ea2"] / 8.314 * (1 / base["T"] - 1 / 573.15))
FA_exact = F0["A"] * np.exp(-k2 * base["P"] * base["Wcat"] / r2["FT_0"])
check("matches analytical solution (k1 = 0)",
      np.isclose(r2["F_out"]["A"], FA_exact, rtol=1e-6))

# 7. Input range guard
try:
    simulate(**{**base, "T": 2000}); check("rejects T out of range", False)
except ValueError:
    check("rejects T out of range", True)

print("\nAll checks passed.")
