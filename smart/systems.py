"""
Reaction systems and a fast batch reactor simulator.

The model is the same packed-bed reactor as reactor_twin.py (isothermal, isobaric, ideal gas,
no pressure drop, plug flow), written so that MANY experiments are solved in one call.  The
smart advisor needs this: to decide on the next experiment it simulates thousands of
candidate operating points for several plausible versions of the chemistry.

Two reaction systems
--------------------
'simple'   the project's current network (identical kinetics to reactor_twin.py)
               A + B -> C          r1 = k1 P_A P_B          (desired)
               A     -> D          r2 = k2 P_A              (side reaction)

'complex'  a more realistic network, typical of hydrogenation / oxidation / alkylation chemistry:
               A + B <=> C         r1 = k1 (P_A P_B - P_C / K_eq(T))     reversible, exothermic
               C + B -> E          r2 = k2 P_C P_B          over-reaction destroys the product
               A     -> D          r3 = k3 P_A              cracking at high temperature
           What makes it hard: hot = fast but the equilibrium and the cracking take product
           away; long contact time / excess B = more conversion but the product reacts on to E.
           The profitable window is narrow and sits in the middle of the box.

Each system has a TRUE kinetics (what the virtual "real reactor" uses, unknown to the
advisor) and a GUESS (an order-of-magnitude starting point a chemist could write down).
The true reactor also has product inhibition (adsorbed C slows every reaction), which the
advisor's power-law model does NOT contain: the advisor's physics is deliberately imperfect,
exactly as it would be on a real reactor.

Kinetic parameters are written per reaction j as
    a_j = ln k_j(T_REF)                 (rate constant at the reference temperature)
    b_j = Ea_j / (R T_REF)              (dimensionless activation energy)
so that  ln k_j(T) = a_j - b_j (T_REF / T - 1).  theta = [a_1, b_1, a_2, b_2, ...].

The equilibrium constant of a reversible reaction comes from thermodynamics (reaction
enthalpy and entropy, which are tabulated), not from fitting:  K_eq = exp(-dH/RT + dS/R).

ALL NUMBERS ARE PLACEHOLDERS, like the rest of the twin.
"""
import numpy as np
from scipy.integrate import solve_ivp

R = 8.314
T_REF = 573.15
P_STD, T_STD = 101325.0, 273.15
MMOL_H = 3.6e6                         # mol/s -> mmol/h


def mfc_to_molar(v_ml_min):
    """Standard mL/min -> mol/s (same as reactor_twin.mfc_to_molar)."""
    return P_STD * (np.asarray(v_ml_min, float) * 1e-6 / 60.0) / (R * T_STD)


def theta_from(k_ref, Ea):
    """(k_ref list, Ea list in J/mol) -> theta vector [a1, b1, a2, b2, ...]."""
    return np.ravel([[np.log(k), E / (R * T_REF)] for k, E in zip(k_ref, Ea)])


def theta_to_physical(theta):
    """theta -> (k_ref array, Ea array in kJ/mol), for printing."""
    th = np.asarray(theta).reshape(-1, 2)
    return np.exp(th[:, 0]), th[:, 1] * R * T_REF / 1000.0


class ReactionSystem:
    """
    species     names, inert last
    nu          stoichiometry (n_species x n_reactions)
    fwd         forward reaction orders (n_reactions x n_species)
    rev         reverse orders for reversible reactions (zeros if irreversible)
    thermo      per reaction (dH J/mol, dS J/mol/K) or None if irreversible
    theta_true  true kinetics of the virtual reactor
    theta_guess a chemist's order-of-magnitude guess (start of the fit)
    K_inhib     true product-inhibition constant on C (1/bar); 0 = none.  Not in the advisor model.
    prices      economics used by the profit objective
    """

    def __init__(self, name, species, nu, fwd, rev, thermo, theta_true, theta_guess,
                 K_inhib, prices, product="C", description=""):
        self.name = name
        self.species = list(species)
        self.ns = len(species)
        self.nu = np.asarray(nu, float)
        self.nr = self.nu.shape[1]
        self.fwd = np.asarray(fwd, float)
        self.rev = np.asarray(rev, float)
        self.thermo = thermo
        self.theta_true = np.asarray(theta_true, float)
        self.theta_guess = np.asarray(theta_guess, float)
        self.K_inhib = K_inhib
        self.prices = prices
        self.iC = self.species.index(product)
        self.description = description

    # ------------------------------------------------------------------ physics
    def log_keq(self, T):
        """ln K_eq(T) for every reaction (n x nr); +inf for irreversible reactions."""
        T = np.asarray(T, float)[:, None]
        out = np.full((T.shape[0], self.nr), np.inf)
        for j, th in enumerate(self.thermo):
            if th is not None:
                dH, dS = th
                out[:, j] = (-dH / (R * T) + dS / R)[:, 0]
        return out

    def rates(self, Pp, T, theta, inhib=0.0):
        """Reaction rates (n x nr) in mol g^-1 s^-1 from partial pressures Pp (n x ns, bar).
        theta is (p,) for one kinetics or (n, p) for one kinetics per row."""
        th = np.atleast_2d(theta)
        a, b = th[:, 0::2], th[:, 1::2]                         # (1 or n, nr)
        lnk = a - b * (T_REF / T[:, None] - 1.0)
        lnP = np.log(np.maximum(Pp, 1e-30))
        fwd = np.exp(lnk + lnP @ self.fwd.T)
        r = fwd.copy()
        lk = self.log_keq(T)
        rev_j = np.isfinite(lk[0])
        if rev_j.any():
            back = np.exp(lnk[:, rev_j] + lnP @ self.rev[rev_j].T - lk[:, rev_j])
            r[:, rev_j] = fwd[:, rev_j] - back
        if inhib:
            r = r / (1.0 + inhib * Pp[:, [self.iC]]) ** 2
        return r

    # ------------------------------------------------------------------ batch simulation
    def simulate(self, P, VA, VB, VI, W, T, theta=None, truth=False, method="fast", groups=None):
        """Outlet molar flows (n x ns, mol/s) for n experiments at once.

        truth=True -> the virtual real reactor (true kinetics + product inhibition).
        Otherwise the advisor's model with the given theta (no inhibition).
        Integrates in the dimensionless bed coordinate s = W / Wcat in [0, 1], with flows
        divided by the inlet total flow, so all states are O(1).

        method='fast'   vectorised adaptive Rosenbrock (ROS2), every row with its own step
                        size; ~10x faster than LSODA for big batches, agrees with it to ~1e-3 of
                        the inlet flow (below GC noise; checked in test_smart.py).  Used for
                        the advisor's internal model, which simulates thousands of candidates.
        method='lsoda'  scipy LSODA on the stacked system, tight tolerances.  Used for the
                        virtual real reactor (one experiment at a time) and as the reference.
        groups          optional integer label per row; rows with the same label take exactly
                        the same integration steps.  Used for finite-difference derivatives
                        with respect to the kinetics (an experiment and its perturbed copies
                        share a label), so the derivatives are smooth."""
        P, VA, VB, VI, W, T = (np.atleast_1d(np.asarray(v, float)) for v in (P, VA, VB, VI, W, T))
        n = max(map(len, (P, VA, VB, VI, W, T)))
        P, VA, VB, VI, W, T = (np.broadcast_to(v, (n,)).copy() for v in (P, VA, VB, VI, W, T))
        if truth:
            theta, inhib = self.theta_true, self.K_inhib
        else:
            inhib = 0.0
        theta = np.asarray(theta, float)
        F0 = np.zeros((n, self.ns))
        F0[:, 0], F0[:, 1], F0[:, -1] = mfc_to_molar(VA), mfc_to_molar(VB), mfc_to_molar(VI)
        FT0 = F0.sum(1)
        f0 = F0 / FT0[:, None]
        scale = W / FT0                                       # g s / mol
        nuT = self.nu.T

        th_rows = theta.ndim == 2

        def g(f, idx=slice(None)):                            # f: (rows, ns) -> df/ds
            f = np.maximum(f, 0.0)
            Pp = P[idx, None] * f / f.sum(1, keepdims=True)
            th = theta[idx] if th_rows else theta
            return scale[idx, None] * (self.rates(Pp, T[idx], th, inhib) @ nuT)

        if method == "lsoda":
            sol = solve_ivp(lambda s, y: g(y.reshape(n, -1)).ravel(), (0.0, 1.0), f0.ravel(),
                            method="LSODA", rtol=1e-8, atol=1e-12,
                            lband=self.ns - 1, uband=self.ns - 1)
            if not sol.success:
                raise RuntimeError(sol.message)
            f = sol.y[:, -1].reshape(n, -1)
        else:
            f = _ros2(g, f0, groups=groups)
        return np.maximum(f, 0.0) * FT0[:, None], F0

    # ------------------------------------------------------------------ what the GC sees
    @staticmethod
    def mole_fractions(F):
        return F / F.sum(1, keepdims=True)

    def flows_from_gc(self, y, VA, VB, VI):
        """Reconstruct outlet molar flows from a GC reading using the inert as internal
        standard (it does not react, so F_I,out = F_I,in).  This is how the operator's GC
        numbers are turned into a mass balance without any extra measurement."""
        y = np.atleast_2d(y)
        FI = mfc_to_molar(VI)
        return y * (FI / y[:, -1])[:, None]

    # ------------------------------------------------------------------ economics
    def profit(self, F, F0, P, W):
        """Process margin in money units per hour (placeholder prices):
             value of C made - cost of A and B fed - disposal of by-products
             - compression energy (feed * ln P) - catalyst cost
        Same structure as 'margin' in bo/problem.py."""
        p = self.prices
        Fm, F0m = F * MMOL_H, F0 * MMOL_H
        byp = sum(Fm[:, self.species.index(s)] * c for s, c in p.get("disposal", {}).items())
        return (p["v_C"] * Fm[:, self.iC]
                - p["c_feed"] * (F0m[:, 0] + F0m[:, 1])
                - byp
                - p["c_comp"] * F0m.sum(1) * np.log(np.asarray(P, float))
                - p["c_cat"] * np.asarray(W, float))

    def kpis(self, F, F0):
        """Conversion of A, selectivity to C, yield of C."""
        FA0 = F0[:, 0]
        XA = (FA0 - F[:, 0]) / FA0
        YC = F[:, self.iC] / FA0
        made = F[:, 2:-1].sum(1)
        SC = np.where(made > 0, F[:, self.iC] / np.maximum(made, 1e-30), np.nan)
        return XA, SC, YC


_GAMMA = 1.0 + 1.0 / np.sqrt(2.0)


def _solve(M, b):
    """Batched M x = b; falls back to least squares if a matrix is singular (only happens
    for absurd trial kinetics, e.g. a wild parameter draw early in a fit)."""
    try:
        return np.linalg.solve(M, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(M) @ np.nan_to_num(b)[..., None])[..., 0]


def _ros2(g, f0, rtol=3e-3, atol=3e-5, h0=1e-9, max_iter=5000, groups=None):
    """Second-order L-stable Rosenbrock method (ROS2, Verwer et al. 1999) with step-size
    control, vectorised over experiments: each row (experiment) has its own step size, and
    every pass of the loop advances all rows that have not reached the outlet yet.

        (I - gamma h J) k1 = g(y)
        (I - gamma h J) k2 = g(y + h k1) - 2 k1
        y_new = y + h (3/2 k1 + 1/2 k2)
        error estimate = y_new - (linearly-implicit Euler step  y + h (I - h J)^-1 g(y))

    L-stability matters here: at high T and P the reactions are up to ~1e7 times faster than
    the flow through the bed, so the first part of the bed is extremely stiff.  The Jacobian
    of each row (6 x 6) is built by finite differences, for all active rows at once."""
    n, ns = f0.shape
    f, s = f0.copy(), np.zeros(n)
    h = np.full(n, h0)
    eye = np.eye(ns)
    for _ in range(max_iter):
        act = np.nonzero(s < 1.0 - 1e-12)[0]
        if len(act) == 0:
            break
        y = f[act]
        hh = np.minimum(h[act], 1.0 - s[act])
        g0 = g(y, act)
        J = np.empty((len(act), ns, ns))
        for i in range(ns):
            d = 1e-7 * np.maximum(np.abs(y[:, i]), 1e-4)
            yp = y.copy()
            yp[:, i] += d
            J[:, :, i] = (g(yp, act) - g0) / d[:, None]
        J[~np.isfinite(J)] = 0.0
        g0 = np.nan_to_num(g0, nan=0.0, posinf=0.0, neginf=0.0)
        M = eye - (_GAMMA * hh)[:, None, None] * J
        k1 = _solve(M, g0)
        k2 = _solve(M, g(y + hh[:, None] * k1, act) - 2.0 * k1)
        y_new = y + hh[:, None] * (1.5 * k1 + 0.5 * k2)
        # error estimate: difference to linearly-implicit Euler, which is also L-stable, so
        # the two agree on stiff components that sit at their equilibrium (ROS2's own embedded
        # first-order formula does not, and forces ~1000 needless steps per row)
        y_eul = y + hh[:, None] * _solve(eye - hh[:, None, None] * J, g0)
        err = np.abs(y_new - y_eul) / (atol + rtol * np.abs(y_new))
        ratio = err.max(1)
        ratio[(y_new < -atol).any(1) | ~np.isfinite(ratio)] = 1e3       # reject negative / NaN steps
        if groups is not None:                       # one step size per group (see simulate)
            gmax = np.zeros(groups.max() + 1)
            np.maximum.at(gmax, groups[act], ratio)
            ratio = gmax[groups[act]]
        ok = ratio <= 1.0
        f[act[ok]] = y_new[ok]
        s[act[ok]] += hh[ok]
        h[act] = hh * np.clip(0.9 / np.sqrt(np.maximum(ratio, 1e-10)), 0.2, 5.0)
    else:
        raise RuntimeError("ROS2 did not reach the reactor outlet")
    return f


# =================================================================================== systems
_BASE_PRICES = dict(v_C=1.0, c_feed=0.20, c_comp=0.03, c_cat=3.0)

SIMPLE = ReactionSystem(
    "simple",
    species=["A", "B", "C", "D", "I"],
    nu=[[-1, -1], [-1, 0], [1, 0], [0, 1], [0, 0]],
    fwd=[[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
    rev=[[0] * 5, [0] * 5],
    thermo=[None, None],
    theta_true=theta_from([2.0e-7, 2.0e-7], [80e3, 120e3]),     # = reactor_twin.DEFAULT_THETA
    theta_guess=theta_from([1e-6, 1e-6], [100e3, 100e3]),
    K_inhib=0.02,
    prices=dict(_BASE_PRICES, disposal={}),
    description="A + B -> C (wanted),  A -> D (side reaction)",
)

COMPLEX = ReactionSystem(
    "complex",
    species=["A", "B", "C", "D", "E", "I"],
    #       r1  r2  r3
    nu=[[-1, 0, -1],     # A
        [-1, -1, 0],     # B
        [1, -1, 0],      # C
        [0, 0, 1],       # D
        [0, 1, 0],       # E
        [0, 0, 0]],      # I
    fwd=[[1, 1, 0, 0, 0, 0],      # r1 forward: P_A P_B
         [0, 1, 1, 0, 0, 0],      # r2: P_B P_C
         [1, 0, 0, 0, 0, 0]],     # r3: P_A
    rev=[[0, 0, 1, 0, 0, 0],      # r1 reverse: P_C / K_eq
         [0] * 6, [0] * 6],
    thermo=[(-120e3, -170.0), None, None],         # A + B <=> C: exothermic, fewer molecules
    theta_true=theta_from([3.0e-6, 3.0e-7, 1.0e-6], [70e3, 120e3, 150e3]),
    theta_guess=theta_from([1e-6, 1e-6, 1e-6], [100e3, 100e3, 100e3]),
    K_inhib=0.03,
    prices=dict(_BASE_PRICES, disposal={"D": 0.10, "E": 0.15}),
    description="A + B <=> C (wanted, reversible),  C + B -> E (over-reaction),  A -> D (cracking)",
)

# Stress test: the advisor's model of the complex reactor with the over-reaction C + B -> E
# MISSING, as if the chemist did not know it happens.  E still shows up in the GC, the model
# just cannot explain it; only the GP correction can account for the lost product.
COMPLEX_MISSING_E = ReactionSystem(
    "complex_missing_E",
    species=COMPLEX.species,
    nu=COMPLEX.nu[:, [0, 2]],
    fwd=COMPLEX.fwd[[0, 2]],
    rev=COMPLEX.rev[[0, 2]],
    thermo=[COMPLEX.thermo[0], None],
    theta_true=COMPLEX.theta_true[[0, 1, 4, 5]],
    theta_guess=COMPLEX.theta_guess[[0, 1, 4, 5]],
    K_inhib=0.0,
    prices=COMPLEX.prices,
    description="advisor model WITHOUT the over-reaction C + B -> E",
)

SYSTEMS = {"simple": SIMPLE, "complex": COMPLEX}
MODELS = {"complex_missing_E": COMPLEX_MISSING_E}

# ------------------------------------------------------------------ the search box (same as bo/problem.py)
KEYS = ["P", "V_A", "V_B", "T", "Wcat"]
NAMES = ["P (bar)", "V_A (mL/min)", "V_B (mL/min)", "T (K)", "Wcat (g)"]
UNITS = ["bar", "mL/min", "mL/min", "K", "g"]
LOW = np.array([1.0, 1.0, 1.0, 450.0, 0.05])
HIGH = np.array([100.0, 50.0, 50.0, 850.0, 5.0])
V_I = 10.0


def unscale(x):
    return LOW + np.clip(np.atleast_2d(x), 0, 1) * (HIGH - LOW)


def scale(phys):
    return (np.atleast_2d(phys) - LOW) / (HIGH - LOW)


def run_points(system, X, theta=None, truth=False):
    """Simulate scaled points X (n x 5).  Returns (profit, F_out, F0, physical inputs)."""
    ph = unscale(X)
    P, VA, VB, T, W = ph.T
    F, F0 = system.simulate(P, VA, VB, V_I, W, T, theta=theta, truth=truth)
    return system.profit(F, F0, P, W), F, F0, ph
