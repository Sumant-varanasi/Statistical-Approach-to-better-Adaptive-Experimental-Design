"""
Digital twin of a gas-phase catalytic fixed-bed reactor (ACOD310).

Implements the model in digital-twin-fixed-bed-reactor.md:
    A + B -> C   (desired),   r1 = k1(T) * PA * PB
    A     -> D   (side),      r2 = k2(T) * PA
    dF_i/dW = sum_j nu_ij r_j, integrated from W = 0 to W = Wcat
Isothermal, isobaric, ideal gas, no pressure drop.

ALL KINETIC PARAMETERS BELOW ARE PLACEHOLDERS, not real chemistry.
They must be replaced with values agreed with Prof. Khare.
"""
import numpy as np
from scipy.integrate import solve_ivp

R = 8.314                 # J mol^-1 K^-1
P_STD = 101325.0          # Pa   -- ASSUMED MFC reference, confirm from instrument spec
T_STD = 273.15            # K    -- ASSUMED MFC reference, confirm from instrument spec
SPECIES = ["A", "B", "C", "D", "I"]

# Operating limits from the spec (Wcat range is our own assumption)
LIMITS = {
    "V_A": (0.0, 50.0),       # mL/min (standard)
    "V_B": (0.0, 50.0),
    "V_I": (0.0, 50.0),
    "Wcat": (0.0, 5.0),       # g  -- ASSUMED, not in spec
    "T": (298.15, 1073.15),   # K
    "P": (1.0, 100.0),        # bar
}

# Placeholder kinetics: theta = [k1_ref, Ea1, k2_ref, Ea2]
T_REF = 573.15  # K
DEFAULT_THETA = {
    "k1_ref": 2.0e-7,   # mol g^-1 s^-1 bar^-2
    "Ea1": 80e3,        # J/mol
    "k2_ref": 2.0e-7,   # mol g^-1 s^-1 bar^-1
    "Ea2": 120e3,       # J/mol  (higher than Ea1 -> side reaction wins at high T)
}

# Stoichiometric matrix nu[species, reaction]
NU = np.array([
    [-1, -1],  # A
    [-1,  0],  # B
    [+1,  0],  # C
    [ 0, +1],  # D
    [ 0,  0],  # I
], dtype=float)


def mfc_to_molar(v_ml_min):
    """Standard volumetric flow (mL/min) -> molar flow (mol/s). F = P_std V / (R T_std)."""
    return P_STD * (v_ml_min * 1e-6 / 60.0) / (R * T_STD)


def arrhenius(k_ref, Ea, T):
    return k_ref * np.exp(-Ea / R * (1.0 / T - 1.0 / T_REF))


def rates(F, T, P, theta):
    """Reaction rates (mol g^-1 s^-1) at one point in the bed."""
    F = np.maximum(F, 0.0)            # guard against tiny negative solver values
    FT = F.sum()
    PA = P * F[0] / FT                # partial pressures in bar
    PB = P * F[1] / FT
    k1 = arrhenius(theta["k1_ref"], theta["Ea1"], T)
    k2 = arrhenius(theta["k2_ref"], theta["Ea2"], T)
    return np.array([k1 * PA * PB, k2 * PA])


def rhs(W, F, T, P, theta):
    return NU @ rates(F, T, P, theta)


def _check_inputs(V_A, V_B, V_I, Wcat, T, P):
    vals = {"V_A": V_A, "V_B": V_B, "V_I": V_I, "Wcat": Wcat, "T": T, "P": P}
    for name, v in vals.items():
        lo, hi = LIMITS[name]
        if not (lo <= v <= hi):
            raise ValueError(f"{name}={v} outside allowed range [{lo}, {hi}]")
    if V_A <= 0:
        raise ValueError("V_A must be > 0 (conversion of A is undefined otherwise)")


def simulate(V_A, V_B, V_I, Wcat, T, P, theta=None, return_profile=False):
    """
    One simulated experiment.
    Inputs: MFC flows (mL/min std), catalyst mass (g), T (K), P (bar).
    Returns dict with outlet flows, outlet mole fractions (simulated GC), X_A, S_C, Y_C.
    """
    theta = theta or DEFAULT_THETA
    _check_inputs(V_A, V_B, V_I, Wcat, T, P)

    F0 = np.array([mfc_to_molar(V_A), mfc_to_molar(V_B), 0.0, 0.0, mfc_to_molar(V_I)])

    if Wcat == 0:
        Fout, profile = F0.copy(), None
    else:
        sol = solve_ivp(rhs, (0.0, Wcat), F0, args=(T, P, theta),
                        method="LSODA", rtol=1e-8, atol=1e-14, dense_output=return_profile)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        Fout = np.maximum(sol.y[:, -1], 0.0)
        profile = sol if return_profile else None

    FT_out = Fout.sum()
    FA0 = F0[0]
    X_A = (FA0 - Fout[0]) / FA0
    prod = Fout[2] + Fout[3]
    S_C = Fout[2] / prod if prod > 0 else np.nan
    Y_C = Fout[2] / FA0

    out = {
        "F0": dict(zip(SPECIES, F0)),
        "F_out": dict(zip(SPECIES, Fout)),
        "FT_0": F0.sum(),
        "FT_out": FT_out,
        "y_out": dict(zip(SPECIES, Fout / FT_out)),     # simulated GC reading
        "X_A": X_A, "S_C": S_C, "Y_C": Y_C,
        "V_out_actual_mL_min": FT_out * R * T / (P * 1e5) * 1e6 * 60,  # at reactor T, P
    }
    if return_profile:
        out["profile"] = profile
    return out


if __name__ == "__main__":
    res = simulate(V_A=20, V_B=20, V_I=10, Wcat=1.0, T=600, P=10)
    print(f"X_A = {res['X_A']:.3f}, S_C = {res['S_C']:.3f}, Y_C = {res['Y_C']:.3f}")
    print("Simulated GC (outlet mole fractions):",
          {k: round(v, 4) for k, v in res["y_out"].items()})
