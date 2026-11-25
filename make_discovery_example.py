"""
Builds the reaction-discovery examples in examples/discovery/ so that every number in them
can be reproduced.

    python make_discovery_example.py              data files + candidate files + WORKED_EXAMPLE.md
                                                  (runs the reaction finder on every file: about 10 min)
    python make_discovery_example.py --data-only  only the data and candidate files (a few seconds)

What it writes
--------------
Example 1, the "mystery" network (for the automatic search; NOT one of the built-in systems)
    mystery_reactions.txt   the reactions the data were calculated from (the answer)
    mystery_exact.csv       GC mole fractions calculated exactly from the reactions and rate laws
    mystery_noisy.csv       the same readings with the GC noise the advisor assumes
Example 2, the complex network (for ranking typed candidates)
    complex_exact.csv, complex_noisy.csv   built the same way from A + B <=> C, C + B -> E, A -> D
    complex_real.csv        the same runs measured on the virtual "real" reactor (VirtualReactor:
                            adsorbed product and reactant block catalyst sites, plus GC noise),
                            i.e. with physics the plain rate laws do not contain
    complex_short_campaign.csv   NOT a kinetic study, and not runs an optimiser chose: the 5 shared
                            screening runs + 3 allowed steps CHOSEN BY HAND, like the start of an
                            optimisation campaign on the virtual "real" reactor, to show what the
                            finder does with few, step-limited runs (the case its ability to start
                            again was developed on: see SHORT_TRIALS for how it does on others)
    candidates_truth.txt, candidates_close.txt, candidates_similar.txt, candidates_far.txt
Example 3, a rate law that does not follow the equation (for the automatic search)
    zero_order_noisy.csv    the same 30 runs calculated from the network of candidates_close.txt
                            taken as the TRUE one (the over-reaction C + B -> E runs at r = k P_C,
                            zero order in B), with GC noise: can the search find the order?
WORKED_EXAMPLE.md          the full calculation of one run by hand, and what the reaction finder
                            returns for every file
finder_results.json         the finder's results in full (for the website and for checking)

The runs (except the short campaign) are a designed KINETIC STUDY: 30 runs spread over a window of the operating box
(2-40 bar, 5-50 mL/min of each gas, 230-410 C; above about 410 C everything is fully converted
and teaches nothing about rates).  They are NOT step-limited like an optimisation campaign: in
a kinetic study the operator deliberately jumps between very different conditions, waiting
for steady state each time.  The catalyst load (1 g, 100 m2/g) is the same in every run.

ALL KINETIC NUMBERS ARE PLACEHOLDERS, like the rest of the twin.
"""
import argparse
import json
import os
import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

warnings.filterwarnings("ignore")
from smart import discovery as D
from smart.systems import (COMPLEX, A_SPEC, R, T_REF, P_STD, T_STD, MAX_STEP, theta_from, scale, mfc_to_molar,
                           screening_runs, base_case)
from smart.kinetics import GC_NOISE_REL, GC_NOISE_ABS, PRIOR_E
from smart.advisor import VirtualReactor
from discover_reactions import report_discovery, report_ranking, report_next_run, strip

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "discovery")
N_RUNS, WCAT = 30, 1.0
WINDOW_LOW = np.array([2.0, 5.0, 5.0, 5.0, 230.0])       # bar, mL/min x 3, deg C
WINDOW_HIGH = np.array([40.0, 50.0, 50.0, 50.0, 410.0])

# ------------------------------------------------------------------ example 1: the mystery network (the answer)
MYSTERY_TEXT = ("A + B <=> C\n"
                "C -> D + B\n"
                "A + 2 B -> E\n")
MYSTERY_K = [2.0e-6, 1.0e-6, 6.0e-8]      # rate constants at 300 C per gram of catalyst (100 m2/g)
MYSTERY_EA = [65e3, 110e3, 85e3]          # activation energies, J/mol
# ------------------------------------------------------------------ example 2: the complex network (systems.COMPLEX)
COMPLEX_K = [3.0e-6, 3.0e-7, 1.0e-6]
COMPLEX_EA = [70e3, 120e3, 150e3]
SEEDS = dict(design=7, mystery_noise=5, complex_noise=3, complex_real=11, short_campaign=1, zero_noise=0)
SHORT = "complex_short_campaign.csv"
# ------------------------------------------------------------------ example 3: a rate law that does not follow the equation
ZERO = "zero_order_noisy.csv"
# Measured on 25 November 2026 and NOT rerun by this script (it would need the git-ignored benchmark cache and about 6
# minutes on 10 cores): discover() with its defaults on the first 8 runs of the ten campaigns of the v5 advisor in
# results/smart_bench/complex (seeds 0 to 9: the 5 shared screening runs, then 3 steps the advisor chose), campaign
# starts that were not used to develop the search.
SHORT_TRIALS = dict(n=10, found=6, one_pass=3, after_restart=3, found_s=(61, 250), missed_s=(249, 348),
                    missed_x_noise=(3.2, 6.6))
# The same day, the search with the UNPRUNED library (reaction_library(thermo_prune=False): 30 forms instead of 22)
UNPRUNED = {"mystery_noisy.csv": "the true network (299 s), not deciding whether A + 2 B -> E runs one way or both ways",
            "complex_noisy.csv": "the true network (61 s), not deciding whether A -> D and B + C -> E run one way or both ways",
            "complex_real.csv": "the true reactions with A -> D written as two-way, D <=> A (197 s), which inside the box "
                                "predicts the same as the one-way form",
            SHORT: "the true reactions with B + C -> E written as two-way (109 s), not deciding whether A -> D and "
                   "B + C -> E run one way or both ways"}
ZERO_TEXT = D.CANDIDATES["close"]         # A + B <=> C;  C + B -> E with r = k P_C (zero order in B);  A -> D
ZERO_K, ZERO_EA = COMPLEX_K, COMPLEX_EA   # the same placeholder numbers as the complex network


def mystery_system():
    return D.build_system(MYSTERY_TEXT, theta=theta_from(MYSTERY_K, MYSTERY_EA), name="mystery")


def complex_system():
    """The complex network with plain rate laws and its true kinetics (same numbers as systems.COMPLEX)."""
    return D.build_system(D.CANDIDATES["truth"], theta=theta_from(COMPLEX_K, COMPLEX_EA), name="complex")


def zero_order_system():
    """The network of candidates_close.txt taken as the truth: the over-reaction C + B -> E runs
    at r = k P_C, whatever the pressure of B (zero order in B)."""
    return D.build_system(ZERO_TEXT, theta=theta_from(ZERO_K, ZERO_EA), name="zero_order")


def study_design(n=N_RUNS, seed=SEEDS["design"]):
    """The kinetic study: n operating points spread evenly (Sobol sequence) over the study
    window, pressure on a logarithmic scale, rounded to what can be set (0.1 bar, 0.5 mL/min,
    1 deg C).  Rows: P bar, V_A, V_B, V_I mL/min, T kelvin, catalyst g."""
    u = qmc.Sobol(5, scramble=True, seed=seed).random(32)[:n]
    lo, hi = WINDOW_LOW.copy(), WINDOW_HIGH.copy()
    lo[0], hi[0] = np.log(lo[0]), np.log(hi[0])
    ph = lo + u * (hi - lo)
    ph[:, 0] = np.round(np.exp(ph[:, 0]), 1)
    ph[:, 1:4] = np.round(ph[:, 1:4] * 2) / 2
    ph[:, 4] = np.round(ph[:, 4]) + 273.15
    return np.column_stack([ph, np.full(n, WCAT)])


def exact_gc(S, phys):
    """GC mole fractions calculated from the reactions and rate laws of S with its true
    kinetics: the mole balances are integrated over the catalyst surface with the accurate
    solver (LSODA, relative tolerance 1e-8).  No adsorption terms, no noise."""
    P, VA, VB, VI, T, W = np.asarray(phys, float).T
    F, _ = S.simulate(P, VA, VB, VI, W, T, theta=S.theta_true, method="lsoda")
    return F / F.sum(1, keepdims=True)


def noisy_gc(y, seed):
    """The GC error model of the advisor: 1 % of the reading + 0.05 mol-% absolute (normal),
    negative readings set to zero, then normalised to 100 %."""
    rng = np.random.default_rng(seed)
    yn = y * (1 + GC_NOISE_REL * rng.standard_normal(y.shape)) + GC_NOISE_ABS * rng.standard_normal(y.shape)
    yn = np.maximum(yn, 0.0)
    return yn / yn.sum(1, keepdims=True)


def real_gc(phys, seed=SEEDS["complex_real"]):
    """The same runs on the virtual 'real' reactor of the project (site blocking by adsorbed C
    and A, which the plain rate laws do not have, and GC noise)."""
    vr = VirtualReactor(COMPLEX, seed=seed, wcat=WCAT, a_spec=A_SPEC)
    return np.array([vr.run(scale(p[:5])[0])["gc"] for p in phys])


def short_campaign(seed=SEEDS["short_campaign"]):
    """Not a kinetic study but the start of an optimisation campaign on the virtual 'real'
    reactor: the five shared screening runs from the project's base case (10 bar, 20 / 20 / 10
    mL/min, 600 K: a temperature ladder, then more pressure, then more B) and three more allowed
    steps towards hotter and higher pressure, chosen by hand (not by an optimiser).  Every run is
    within MAX_STEP of the one before."""
    X = screening_runs(base_case(0))
    X = np.vstack([X, X[-1] + [5, 0, 0, 0, 20], X[-1] + [10, 5, 0, 0, 40], X[-1] + [10, 10, -5, 0, 60]])
    assert np.all(np.abs(np.diff(X, axis=0)) <= MAX_STEP + 1e-9)
    vr = VirtualReactor(COMPLEX, seed=seed, wcat=WCAT, a_spec=A_SPEC)
    return D.make_data(np.column_stack([X, np.full(len(X), WCAT)]), np.array([vr.run(scale(x)[0])["gc"] for x in X]), name=SHORT)


def datasets():
    """{file name: data} for the seven CSV files."""
    phys = study_design()
    ym, yc = exact_gc(mystery_system(), phys), exact_gc(complex_system(), phys)
    return {ZERO: D.make_data(phys, noisy_gc(exact_gc(zero_order_system(), phys), SEEDS["zero_noise"]), name=ZERO),
            "mystery_exact.csv": D.make_data(phys, ym, name="mystery_exact.csv"),
            "mystery_noisy.csv": D.make_data(phys, noisy_gc(ym, SEEDS["mystery_noise"]), name="mystery_noisy.csv"),
            "complex_exact.csv": D.make_data(phys, yc, name="complex_exact.csv"),
            "complex_noisy.csv": D.make_data(phys, noisy_gc(yc, SEEDS["complex_noise"]), name="complex_noisy.csv"),
            "complex_real.csv": D.make_data(phys, real_gc(phys), name="complex_real.csv"),
            SHORT: short_campaign()}


def write_data():
    os.makedirs(OUT, exist_ok=True)
    data = datasets()
    for name, d in data.items():
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            fh.write(D.data_to_csv(d))
    for name, text in D.CANDIDATES.items():
        with open(os.path.join(OUT, f"candidates_{name}.txt"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(OUT, "mystery_reactions.txt"), "w", newline="\n", encoding="utf-8") as fh:
        fh.write("# The reactions mystery_exact.csv and mystery_noisy.csv were calculated from (the answer).\n"
                 "# Rate constants at 300 C per gram of catalyst (100 m2/g) and activation energies:\n"
                 + "".join(f"#   k{j} = {k:g}, Ea{j} = {e / 1000:g} kJ/mol\n" for j, (k, e) in enumerate(zip(MYSTERY_K, MYSTERY_EA), 1))
                 + MYSTERY_TEXT)
    return data


# =================================================================================== one run, step by step
def bed_profile(S, row, areas):
    """Molar flows (mol/s) at catalyst areas `areas` (m2) along the bed for one run, by direct
    integration of dF_i/dS = sum_j nu_ij r_j (independent of the simulator used for the files)."""
    P, VA, VB, VI, T, W = row
    F0 = S.inlet_flows(VA, VB, VI)[0]

    def rhs(_, F):
        F = np.maximum(F, 0.0)
        return S.nu @ S.rates((P * F / F.sum())[None, :], np.array([T]), S.theta_true)[0]

    sol = solve_ivp(rhs, (0.0, areas[-1]), F0, method="LSODA", rtol=1e-10, atol=1e-18, t_eval=areas)
    return sol.y.T


def worked_run(S, data):
    """Index of the run used for the hand calculation: every product clearly visible and the
    conversion of A closest to 50 %."""
    y, ph = data["gc"], data["phys"]
    F = S.flows_from_gc(y, ph[:, 1], ph[:, 2], ph[:, 3])
    X = 1 - F[:, 0] / mfc_to_molar(ph[:, 1])
    ok = (y[:, 2:5] > 0.01).all(1)
    return int(np.argmin(np.where(ok, np.abs(X - 0.5), 9.0)))


def sci(v, digits=4):
    """A rate in scientific notation; exactly '0' when there is nothing to react."""
    return "0" if abs(v) < 1e-25 else f"{v:.{digits}e}"


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def worked_example_md(data):
    S, d = mystery_system(), data["mystery_exact.csv"]
    i = worked_run(S, d)
    P, VA, VB, VI, T, W = d["phys"][i]
    sp = S.species
    net = D.parse_network(MYSTERY_TEXT)
    area = A_SPEC * W
    F0 = S.inlet_flows(VA, VB, VI)[0]
    y0 = F0 / F0.sum()
    k_m2 = np.array(MYSTERY_K) / A_SPEC
    kT = k_m2 * np.exp(-np.array(MYSTERY_EA) / R * (1 / T - 1 / T_REF))
    dH, dS = net[0].thermo()
    K1 = np.exp(-dH / (R * T) + dS / R)
    r0 = S.rates((P * y0)[None, :], np.array([T]), S.theta_true)[0]
    areas = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]) * area / 100.0
    prof = bed_profile(S, d["phys"][i], areas)
    Fout = prof[-1]
    yout = Fout / Fout.sum()
    euler = F0 + areas[1] * (S.nu @ r0)
    u = 1e6                                                   # mol/s -> micromol/s
    L = []
    L += ["# Reaction discovery: worked example", "",
          "*Generated by `python make_discovery_example.py`; every number below is computed, none is typed in. "
          "All kinetic numbers are placeholders, like the rest of the project.*", "",
          "This folder shows that the advisor can work out **which reactions happen** from the GC readings, "
          "in two ways:", "",
          "1. **Automatic search** (example 1, the \"mystery\" data): the program is given only the run log and "
          "finds the reactions and their rate laws.",
          "2. **Ranking candidates** (example 2, the complex data): you type a few possible reaction networks and "
          "the program says which one the data support, for a candidate that is very different from the truth, "
          "one that is relatively the same and one that is quite close.", "",
          "Part A shows how the data files were calculated from the reactions, with one run done step by step. "
          "Part B shows what the reaction finder returns for every file.", "",
          "## Part A. How the data were calculated", "",
          "### A1. The reactor model", "",
          "The reactor is the fixed bed of the project: plug flow, constant temperature and pressure, ideal gas. "
          "The gases react on the catalyst surface, so the bed is followed along the catalyst surface area "
          "S (m2), from 0 at the inlet to S_total = (BET area) x (catalyst mass) at the outlet. For every species i "
          "the mole balance is", "",
          "    dF_i/dS = sum over reactions j of  nu_ij * r_j", "",
          "- F_i: molar flow of species i (mol/s)",
          "- nu_ij: how many molecules of i reaction j makes (+) or uses (-)",
          "- r_j: rate of reaction j per m2 of catalyst (mol m^-2 s^-1), a function of the partial pressures",
          "- partial pressure of i: P_i = P x F_i / (sum of all F), in bar; the inert counts in the sum, so it dilutes",
          "- rate constant at temperature T: k_j(T) = k_j(300 C) x exp[-(Ea_j / R) x (1/T - 1/573.15 K)]  (Arrhenius)",
          "- a reaction written with <=> runs both ways and stops at the equilibrium: its rate has the reverse term "
          "divided by the equilibrium constant K(T) = exp(-dH/(R T) + dS/R), from tabulated reaction enthalpy and entropy",
          "- what the GC reports is the mole fraction of each species at the outlet, y_i = F_i / (sum of all F)", "",
          "Species: A and B are the feeds, I is the inert gas, C = AB, D = A rearranged (one A unit), E = AB2. "
          "Every reaction below keeps the number of A units and B units (atom balance).", "",
          "### A2. Example 1: the mystery network (file `mystery_reactions.txt`)", ""]
    L += md_table(["#", "reaction", "rate law", "k at 300 C per gram", "per m2 (k / 100 m2/g)", "Ea (kJ/mol)"],
                  [[str(j), f"`{r.equation()}`", f"`{r.rate_law(j)}`", f"{MYSTERY_K[j - 1]:.2e} {r.k_units()}",
                    f"{k_m2[j - 1]:.2e}", f"{MYSTERY_EA[j - 1] / 1000:g}"] for j, r in enumerate(net, 1)])
    L += ["", f"Reaction 1 is reversible with dH = {dH / 1000:g} kJ/mol and dS = {dS:g} J/mol/K (it releases heat, so the "
          "equilibrium moves back to A + B when hotter). Reactions 2 and 3 run one way. (That they run one way is part of "
          "how this made-up example is defined. The program's own placeholder table of reaction enthalpies and entropies "
          "would let both run backwards to some extent at some conditions, so the search in part B is also offered their "
          "two-way forms and has to find out from the data which form was used.)", "",
          "The mole balances written out (the same for every run):", "",
          "    dF_A/dS = -r1 - r3          dF_B/dS = -r1 + r2 - 2 r3",
          "    dF_C/dS = +r1 - r2          dF_D/dS = +r2",
          "    dF_E/dS = +r3               dF_I/dS = 0", "",
          f"### A3. One run, step by step: run {i + 1} of `mystery_exact.csv`", "",
          f"**Set points** (row {i + 1} of the file): pressure {P:g} bar, flows {VA:g} / {VB:g} / {VI:g} mL/min of A / B / inert "
          f"(at standard conditions), temperature {T - 273.15:g} C = {T:.2f} K, catalyst {W:g} g with {A_SPEC:g} m2/g, "
          f"so S_total = {area:g} m2.", "",
          "**Step 1. Flows into the bed.** A mass-flow controller set to V mL/min (standard: 0 C, 1.01325 bar) delivers "
          "F = P_std V / (R T_std):", "",
          f"    F = {P_STD:g} Pa x (V x 1e-6 / 60) m3/s / ({R:g} x {T_STD:g} K) = V x {mfc_to_molar(1.0) * u:.5f} micromol/s per mL/min", ""]
    L += md_table(["", "A", "B", "inert", "total"],
                  [["flow in (micromol/s)", f"{F0[0] * u:.4f}", f"{F0[1] * u:.4f}", f"{F0[-1] * u:.4f}", f"{F0.sum() * u:.4f}"],
                   ["mole fraction", f"{y0[0]:.5f}", f"{y0[1]:.5f}", f"{y0[-1]:.5f}", "1"],
                   [f"partial pressure (bar) = {P:g} x fraction", f"{P * y0[0]:.4f}", f"{P * y0[1]:.4f}", f"{P * y0[-1]:.4f}", f"{P:g}"]])
    L += ["", f"**Step 2. Rate constants at {T:.2f} K.** k_j(T) = k_j(300 C) x exp[-(Ea_j / 8.314) x (1/{T:.2f} - 1/573.15)]:", ""]
    L += md_table(["", "k1", "k2", "k3"],
                  [["per m2 at 300 C"] + [f"{v:.3e}" for v in k_m2],
                   ["Arrhenius factor"] + [f"{v:.4f}" for v in kT / k_m2],
                   [f"per m2 at {T - 273.15:g} C"] + [f"{v:.4e}" for v in kT]])
    L += ["", f"Equilibrium constant of reaction 1: K1 = exp(-({dH:g}) / (8.314 x {T:.2f}) + ({dS:g}) / 8.314) = {K1:.4g} per bar.", "",
          "**Step 3. Rates at the inlet (S = 0)**, where there is no product yet:", "",
          f"    r1 = k1 (P_A P_B - P_C / K1) = {kT[0]:.4e} x ({P * y0[0]:.4f} x {P * y0[1]:.4f} - 0) = {r0[0]:.4e} mol m^-2 s^-1",
          f"    r2 = k2 P_C                  = {kT[1]:.4e} x 0 = {sci(r0[1])}",
          f"    r3 = k3 P_A P_B^2            = {kT[2]:.4e} x {P * y0[0]:.4f} x {P * y0[1]:.4f}^2 = {r0[2]:.4e}", "",
          f"So at the inlet A disappears at dF_A/dS = -r1 - r3 = {-(r0[0] + r0[2]) * u:.5f} micromol/s per m2. "
          f"A first small step of {areas[1]:g} m2 with these inlet rates (Euler's method) gives "
          f"F_A = {F0[0] * u:.4f} - {areas[1]:g} x {(r0[0] + r0[2]) * u:.5f} = {euler[0] * u:.4f} micromol/s "
          f"and F_C = {euler[2] * u:.4f}; the accurate integration below gives {prof[1, 0] * u:.4f} and {prof[1, 2] * u:.4f} "
          "(slightly different, because the rates change as soon as A and B are used up and C appears, which is why the "
          "balances are integrated with a proper solver in small adaptive steps).", "",
          "**Step 4. Integration along the bed** (molar flows in micromol/s; scipy LSODA, relative tolerance 1e-10):", ""]
    rows = []
    for a, F in zip(areas, prof):
        Pp = P * F / F.sum()
        r = S.rates(Pp[None, :], np.array([T]), S.theta_true)[0]
        rows.append([f"{a:g}"] + [f"{v * u:.4f}" for v in F[:5]] + [f"{F.sum() * u:.4f}"] + [sci(v, 3) for v in r])
    L += md_table(["S (m2)", "F_A", "F_B", "F_C", "F_D", "F_E", "total", "r1", "r2", "r3"], rows)
    L += ["", "The total flow falls because A + B -> C and A + 2 B -> E join molecules. C is made first and then partly "
          "falls apart into D + B (reaction 2), which gives back the B.", "",
          "**Step 5. What the GC reads at the outlet** (mole fraction = flow / total flow):", ""]
    csv_row = d["gc"][i]
    L += md_table([""] + sp,
                  [["flow out (micromol/s)"] + [f"{v * u:.4f}" for v in Fout],
                   ["mole fraction, this calculation"] + [f"{v:.6f}" for v in yout],
                   [f"row {i + 1} of `mystery_exact.csv`"] + [f"{v:.6f}" for v in csv_row],
                   [f"row {i + 1} of `mystery_noisy.csv` (with GC noise)"] + [f"{v:.6f}" for v in data["mystery_noisy.csv"]["gc"][i]]])
    atoms_in = (F0[0], F0[1])
    atoms_out = (Fout[0] + Fout[2] + Fout[3] + Fout[4], Fout[1] + Fout[2] + 2 * Fout[4])
    L += ["", f"The hand calculation and the file agree: the largest difference is {np.max(np.abs(yout - np.round(csv_row, 6))):.1e}, "
          "which is the rounding of the file to 6 decimals. (The file was made by the project's simulator, which "
          "integrates all 30 runs in one call with a relative tolerance of 1e-8; step 4 integrates this one run on its own "
          "with 1e-10. `test_discovery.py` also recalculates rows of both exact files from the rate laws written out by "
          "hand, without using the project's simulator at all.)", "",
          f"**Check: atom balance.** A units in = {atoms_in[0] * u:.4f}, out (A + C + D + E) = {atoms_out[0] * u:.4f} micromol/s; "
          f"B units in = {atoms_in[1] * u:.4f}, out (B + C + 2 E) = {atoms_out[1] * u:.4f} micromol/s. "
          f"Conversion of A = 1 - {Fout[0] * u:.4f} / {F0[0] * u:.4f} = {100 * (1 - Fout[0] / F0[0]):.1f} %.", "",
          "**The noisy file.** Each exact mole fraction y is replaced by y x (1 + 0.01 z1) + 0.0005 z2, with z1 and z2 "
          "random numbers from a standard normal distribution (1 % of the reading plus 0.05 mol-% absolute: the "
          "repeatability of an online GC). Negative results are set to zero and the row is rescaled to sum to 1.", "",
          "### A4. Example 2: the complex network (`candidates_truth.txt`)", ""]
    cnet = D.parse_network(D.CANDIDATES["truth"])
    L += md_table(["#", "reaction", "rate law", "k at 300 C per gram", "Ea (kJ/mol)"],
                  [[str(j), f"`{r.equation()}`", f"`{r.rate_law(j)}`", f"{COMPLEX_K[j - 1]:.2e} {r.k_units()}",
                    f"{COMPLEX_EA[j - 1] / 1000:g}"] for j, r in enumerate(cnet, 1)])
    L += ["", "`complex_exact.csv` and `complex_noisy.csv` are calculated exactly as above (same 30 runs, same steps), "
          "with these reactions. `complex_real.csv` holds the same runs measured on the project's virtual \"real\" reactor: "
          "there every rate is also divided by (1 + K_C P_C + K_A P_A)^m (m = number of molecules in the reaction step), "
          "because adsorbed product C and reactant A occupy part of the catalyst surface (K_C = 0.03 per bar at 300 C, "
          "falling when hotter; K_A = 0.005 per bar), and the GC noise is added. The plain rate laws do not contain "
          "this, so the file shows what happens when the physics is imperfect.", "",
          "### A5. The 30 runs", "",
          "A designed kinetic study: 30 operating points spread evenly over 2-40 bar, 5-50 mL/min of each gas and "
          "230-410 C, the same for every file, with 1 g of catalyst. They are **not** step-limited like an optimisation "
          "campaign: a kinetic study deliberately jumps between very different conditions. Conversion of A in the "
          f"mystery file ranges from {100 * _conv(S, d).min():.0f} % to {100 * _conv(S, d).max():.0f} %.", "",
          f"Two more files: `{SHORT}` is deliberately different, {len(data[SHORT]['gc'])} step-limited runs from the start of an "
          f"optimisation campaign on the virtual \"real\" reactor (see B5); `{ZERO}` holds the same 30 runs calculated from a "
          "network with a rate law that does not follow its equation (see B6).", ""]
    return L


def _conv(S, d):
    F = S.flows_from_gc(d["gc"], d["phys"][:, 1], d["phys"][:, 2], d["phys"][:, 3])
    return 1 - F[:, 0] / mfc_to_molar(d["phys"][:, 1])


# =================================================================================== the reaction finder on the files
def run_finder(data, log=print):
    """Every result shown in part B of WORKED_EXAMPLE.md."""
    res, t0 = {}, time.time()

    def note(msg):
        log(f"   [{time.time() - t0:5.0f} s] {msg}")

    for name in ("mystery_noisy.csv", "mystery_exact.csv", "complex_real.csv"):
        note(f"automatic search on {name}")
        res[f"discover:{name}"] = D.discover(data[name])
    for name in ("complex_exact.csv", "complex_noisy.csv", "complex_real.csv"):
        note(f"ranking the four candidates on {name}")
        res[f"rank:{name}"] = D.rank_candidates(D.CANDIDATES, data[name])
    for start in ("far", "similar", "close"):
        note(f"repairing candidate '{start}' on complex_noisy.csv")
        res[f"repair:{start}"] = D.discover(data["complex_noisy.csv"], start=D.CANDIDATES[start])
    note(f"automatic search and ranking on {SHORT}")
    res[f"discover:{SHORT}"] = D.discover(data[SHORT])
    res[f"rank:{SHORT}"] = D.rank_candidates(D.CANDIDATES, data[SHORT])
    note(f"automatic search on {ZERO}")
    res[f"discover:{ZERO}"] = D.discover(data[ZERO])
    note("runs that tell networks apart")
    d = data["complex_noisy.csv"]
    res["next:complex_free"] = D.discriminating_run(res["rank:complex_noisy.csv"], d)
    res["next:complex_step"] = D.discriminating_run(res["rank:complex_noisy.csv"], d, x_last=d["phys"][-1, :5])
    res["next:mystery_free"] = D.discriminating_run(res["discover:mystery_noisy.csv"], data["mystery_noisy.csv"])
    note("done")
    return res


def same_network(a, b):
    """True if two networks (text or lines) hold exactly the same reactions and rate laws."""
    return D.network_key(D.parse_network(a)) == D.network_key(D.parse_network(b))


def solver_gap(found, nr):
    """How far the fast bed solver (used for every fit) and the accurate one (LSODA) differ at a proposed run, for
    the two networks the run is meant to separate: the largest difference of a predicted GC reading, in mol-%."""
    x = np.r_[np.asarray(nr["x"], float), nr["run"]["catalyst_g"]]
    worst = (0.0, "")
    for f in [found["best"]] + list(found["runners_up"][:1]):
        kf = D._model_of(f)
        out = {}
        for meth in ("fast", "lsoda"):
            F, _ = kf.simulate(x[None, :], kf.theta) if meth == "fast" else (None, None)
            if meth == "lsoda":
                P, VA, VB, VI, T, W = x
                parts = kf.split(kf.theta)
                K = None
                if "lnK" in parts:
                    K = np.exp(parts["lnK"] + (parts["e"] if "e" in parts else PRIOR_E) * (T_REF / T - 1.0))
                F, _ = kf.S.simulate(P, VA, VB, VI, W, T, theta=parts["kin"], method="lsoda", a_spec=kf.a_spec, inhib=K)
            out[meth] = 100 * F[0] / F[0].sum()
        d = np.abs(out["fast"] - out["lsoda"])[:-1]
        i = int(np.argmax(d))
        if d[i] > worst[0]:
            worst = (float(d[i]), f"{COMPLEX.species[i]} {out['fast'][i]:.1f} against {out['lsoda'][i]:.1f} mol-% for "
                                  f"'{'; '.join(f['reactions'])}'")
    return dict(mol_pct=worst[0], what=worst[1], where=f"{x[0]:g} bar, {x[4] - 273.15:.0f} C")


def finder_md(res):
    def block(lines):
        return ["```"] + lines + ["```", ""]

    my, mx, cr = res["discover:mystery_noisy.csv"], res["discover:mystery_exact.csv"], res["discover:complex_real.csv"]
    gap = solver_gap(my, res["next:mystery_free"])
    lib = D.reaction_library()
    n_var = sum(len(e["variants"]) for e in lib)
    L = ["## Part B. What the reaction finder returns", "",
         "How it decides, in short: every network tried is fitted to all GC readings (rate constants and activation "
         "energies). The misfit is counted in units of the GC noise, and networks are compared by BIC = misfit + a "
         "penalty for every fitted number, so an extra reaction must clearly improve the fit to be kept. A BIC "
         "difference below 6 is reported as *the data cannot tell these apart*.", "",
         "### B1. Automatic search on the mystery data", "",
         "    python discover_reactions.py examples/discovery/mystery_noisy.csv", "",
         "The search starts with no reactions and may use any reaction from a library that is **enumerated from the "
         f"atom balance** ({len(lib)} reactions among A, B, C, D, E with at most 3 molecules reacting and 2 formed; "
         f"{n_var} forms in all: one-way in either direction and reversible, except where the placeholder thermodynamics "
         "says a direction runs to completion everywhere in the operating box, and then only that direction is offered), "
         "not written around the answer. Each step of the search tries adding a reaction, removing one, changing one "
         "reaction order and, if none of these helps, replacing a reaction.", "",
         "One caution about that pruning. The placeholder thermodynamics (the species table in `smart/discovery.py`) "
         "were chosen to reproduce the complex network, with its two one-way steps (A -> D and C + B -> E) complete "
         "everywhere in the box; for that chemistry the pruning therefore removes exactly the other forms of those two "
         "reactions. It encodes the directions of the complex example's answer, and making A -> D complete needs an "
         "entropy for D that an isomerisation would hardly have. Without pruning (30 forms) the search returns on this "
         f"file {UNPRUNED['mystery_noisy.csv']}; on the complex files see B8.", ""]
    L += block(report_discovery(my, "mystery_noisy.csv"))
    truth = MYSTERY_TEXT
    hit = same_network(my["best"]["reactions"], truth)
    tied_hit = any(same_network(f["reactions"], truth) and f["tied_with_best"] for f in my["runners_up"])
    L += ["**Compared with the answer** (`mystery_reactions.txt`: " + "; ".join(D.network_text(truth)) + "): "
          + ("the search returns exactly these reactions and rate laws." if hit else
             "the true network is among the networks the search reports as tied with its best." if tied_hit else
             "the search did NOT return the true network; see the lists above for what it found instead.")]
    if my["undecided"]:
        L += ["What the data leave open, as reported by the search: " + " ".join(my["undecided"])]
    L += ["", "Fitted against true kinetics:", ""]
    rows = []
    for j, r in enumerate(D.parse_network(truth)):
        f = next((q for q in my["best"]["rate_constants"] if D.parse_reaction(q["reaction"]).direction_key() == r.direction_key()), None)
        rows.append([f"`{r.equation()}`", f"{MYSTERY_K[j]:.2e}", f"{f['k_ref']:.2e}" if f else "not found",
                     f"{MYSTERY_EA[j] / 1000:g}", f"{f['Ea_kJ']:.1f} +- {f['Ea_sd_kJ']:.1f}" if f else "not found"])
    L += md_table(["reaction", "true k (300 C)", "fitted k", "true Ea (kJ/mol)", "fitted Ea"], rows)
    L += ["", f"On the noise-free file `mystery_exact.csv` the search returns: {'; '.join(mx['best']['reactions'])} "
          f"({mx['best']['verdict']}; {'the true network' if same_network(mx['best']['reactions'], truth) else 'not the true network'}"
          + (f"; left open: {' '.join(mx['undecided'])}" if mx["undecided"] else "") + f"). {mx['n_fits']} fits, {mx['seconds']:.0f} s.", "",
          "The run that would settle what is left open (`--next-run`):", ""]
    L += block(report_next_run(res["next:mystery_free"]))
    L += ["### B2. Ranking the three candidates on the complex data", "",
          "    python discover_reactions.py examples/discovery/complex_noisy.csv --candidates examples/discovery/candidates_truth.txt "
          "examples/discovery/candidates_close.txt examples/discovery/candidates_similar.txt examples/discovery/candidates_far.txt", "",
          "The candidates (text files in this folder):", ""]
    L += md_table(["candidate", "what it is", "reactions and rate laws"],
                  [[f"`{n}`", D.CANDIDATE_WORDS[n], "<br>".join(f"`{e}`" for e in D.format_network(D.CANDIDATES[n]))] for n in D.CANDIDATES])
    L += [""]
    for name, title in (("complex_exact.csv", "Exact data (no noise)"), ("complex_noisy.csv", "With GC noise"),
                        ("complex_real.csv", "The \"real\" reactor (site blocking the plain rate laws do not have, plus GC noise)")):
        L += [f"**{title}: `{name}`**", ""] + block(report_ranking(res[f"rank:{name}"], name))
    rr = res["rank:complex_real.csv"]["ranking"][0]
    L += [f"On the realistic file the best candidate is '{rr['name']}', fitted with {rr['model_words']}. "
          "The program is allowed one extra number for product C blocking catalyst sites and keeps it only when the data "
          "ask for it (they do here, and not on the two files made with plain rate laws). The reactant term K_A of the "
          "virtual reactor is not in the model at all; its effect is absorbed by the fitted constants. One thing makes "
          "this file a favourable case: how K_C falls with temperature is not fitted but assumed (a typical heat of "
          f"adsorption, {-PRIOR_E * R * T_REF / 1000:.0f} kJ/mol), and the virtual reactor happens to use the same value.", "",
          "The run that would best tell the first two candidates apart, anywhere in the operating box and within one "
          "allowed step of the last run of the log:", ""]
    L += block(report_next_run(res["next:complex_free"]) + [""] + report_next_run(res["next:complex_step"]))
    L += ["The difference between the two predictions is given twice: against the GC noise alone, and against the GC noise "
          "plus the uncertainty of each network's own prediction (its fitted rate constants are not exact, least of all far "
          "from the runs done). The second, smaller number is the one to plan with; the expected gain in BIC is based on it.", ""]
    L += ["### B3. Repairing a wrong candidate", "",
          "    python discover_reactions.py examples/discovery/complex_noisy.csv --start examples/discovery/candidates_far.txt", "",
          "Starting from each wrong candidate, the search adds, removes and replaces reactions and re-checks the orders:", ""]
    ctruth = D.CANDIDATES["truth"]
    for start in ("far", "similar", "close"):
        r = res[f"repair:{start}"]
        end = ("ends at the true network" if same_network(r["best"]["reactions"], ctruth) else
               "ends at a network tied with the true one" if any(same_network(f["reactions"], ctruth) and f["tied_with_best"] for f in r["runners_up"])
               else "does NOT end at the true network")
        L += [f"**From `{start}`** ({D.CANDIDATE_WORDS[start]}): {end} ({r['n_fits']} fits, {r['seconds']:.0f} s).", ""]
        L += block(r["path"] + [""] + r["summary"])
    L += ["### B4. Automatic search on the realistic file", "",
          "    python discover_reactions.py examples/discovery/complex_real.csv", ""]
    L += block(report_discovery(cr, "complex_real.csv"))
    L += ["With plain rate laws alone no network of the library fits this file within the GC noise, because the "
          "real reactor has site blocking. " + ("The search ends at the true reactions with site blocking."
          if same_network(cr["best"]["reactions"], ctruth) else "The search does NOT end at the true reactions here; "
          "see the output above."), ""]
    sc, sr = res[f"discover:{SHORT}"], res[f"rank:{SHORT}"]
    sc_hit = same_network(sc["best"]["reactions"], ctruth)
    L += ["### B5. Few, step-limited runs: the start of an optimisation campaign", "",
          f"    python discover_reactions.py examples/discovery/{SHORT}", "",
          f"`{SHORT}` is not a kinetic study, and its steps were not chosen by an optimiser. It holds {sc['best']['n_runs']} "
          "runs on the virtual \"real\" reactor, like the start of an optimisation campaign: the 5 shared screening runs "
          "from the base case (10 bar, 20 / 20 / 10 mL/min, 600 K: a temperature ladder, then more pressure, then more B) "
          "and 3 more steps to hotter and higher pressure CHOSEN BY HAND, each within the allowed step (10 bar, 10 mL/min, "
          "30 K) of the run before. It is the case on which the search's ability to start again was developed: the search "
          "changes one reaction at a time, and here its first choice leads to a dead end. It notices that its result does "
          "not fit within the GC noise and starts again from another first reaction. So this is a hand-picked, favourable "
          "example, not a typical one.", "",
          f"How the search does on campaign starts it was not developed on was measured separately (not shipped as files): "
          f"on the first 8 runs of {SHORT_TRIALS['n']} step-limited campaigns of the v5 advisor on the same virtual reactor "
          "(the shared screening runs, then 3 steps the advisor chose; seeds 0 to 9) it found the true network in "
          f"{SHORT_TRIALS['found']} of {SHORT_TRIALS['n']} ({SHORT_TRIALS['one_pass']} in one pass, {SHORT_TRIALS['after_restart']} "
          f"after starting again; {SHORT_TRIALS['found_s'][0]} to {SHORT_TRIALS['found_s'][1]} s). In the other "
          f"{SHORT_TRIALS['n'] - SHORT_TRIALS['found']} it ended, after three new starts and {SHORT_TRIALS['missed_s'][0]} to "
          f"{SHORT_TRIALS['missed_s'][1]} s, at a wrong network that it reported as not fitting within the GC noise (typical "
          f"miss {SHORT_TRIALS['missed_x_noise'][0]:g} to {SHORT_TRIALS['missed_x_noise'][1]:g} times the noise), never as an "
          "answer. Ten campaign starts are a small sample.", ""]
    L += block(report_discovery(sc, SHORT))
    L += [("The search ends at the true reactions" if sc_hit else "The search does NOT end at the true reactions")
          + f" after starting again {sc['restarts']} time{'s' if sc['restarts'] != 1 else ''}. "
          + ("With 8 runs the fit is less sharp than with the 30-run study: compare the uncertainties of the fitted numbers above "
             "with those in B2. " if sc_hit else "")
          + ("What these 8 runs leave open, as reported by the search: " + " ".join(sc["undecided"]) if sc["undecided"] else
             "The next best networks are ruled out, but by much smaller BIC differences than in the 30-run study."), "",
          "Ranking the four typed candidates on the same runs:", ""]
    L += block(report_ranking(sr, SHORT))
    zo = res[f"discover:{ZERO}"]
    zo_hit = same_network(zo["best"]["reactions"], ZERO_TEXT)
    zo_tied = any(same_network(f["reactions"], ZERO_TEXT) and f["tied_with_best"] for f in zo["runners_up"])
    L += ["### B6. A rate law that does not follow the equation (order of reaction)", "",
          f"    python discover_reactions.py examples/discovery/{ZERO}", "",
          f"`{ZERO}` holds the same 30 runs calculated from the network of `candidates_close.txt` taken as the TRUE one:", ""]
    L += block(D.format_network(ZERO_TEXT))
    L += ["Reaction 2 uses up B but its rate does not depend on the pressure of B (zero order in B: the catalyst surface "
          "is saturated with B). So that B cannot go below zero, the data were calculated with the rate multiplied by "
          f"y_B / (y_B + {D.ZERO_ORDER_CUT:g}), y_B being the mole fraction of B: 1 for all practical purposes, falling to zero "
          "when B has run out. Kinetic numbers as for the complex network (A4); GC noise as in the other noisy files. "
          "The search is not told that any order is unusual:", ""]
    L += block(report_discovery(zo, ZERO))
    L += [("The search returns exactly the true reactions and rate laws, including the zero order in B." if zo_hit else
           "The true network is among the networks the search reports as tied with its best." if zo_tied else
           "The search did NOT return the true network here; see the output above for what it found instead.")
          + " A rate law that does not follow its equation is the hardest case for the search: it takes every rate to "
          "follow its equation until the reactions are settled, and first tries to make up for the unusual order with extra "
          f"reactions. This file is a favourable case ({'one pass' if not zo['restarts'] else 'see the output above'}). "
          "In six further trials, not shipped as files, the same network was used with other noise on these 30 runs (2 "
          "trials: found in one pass, as here) and with four other run designs. On those four the first pass and one new "
          "start found it once and missed it three times, each miss ending with *does not fit within the GC noise* rather "
          "than with a wrong answer presented as right. From its second new start on, the search therefore also tries every "
          "reaction it may add with one of its orders changed; with that it found the true network in all four, after one "
          "or two new starts and 2 to 3 minutes instead of about 1 (see B8).", ""]
    L += ["### B7. Summary", ""]
    rows = [["mystery_noisy.csv", "automatic search", "; ".join(my["best"]["reactions"]), "yes" if hit else ("tied" if tied_hit else "NO"),
             f"{my['seconds']:.0f} s"],
            ["mystery_exact.csv", "automatic search", "; ".join(mx["best"]["reactions"]),
             "yes" if same_network(mx["best"]["reactions"], truth) else "see text", f"{mx['seconds']:.0f} s"],
            ["complex_real.csv", "automatic search", "; ".join(cr["best"]["reactions"]) + f" ({cr['best']['model']})",
             "yes" if same_network(cr["best"]["reactions"], ctruth) else "NO", f"{cr['seconds']:.0f} s"]]
    for name in ("complex_exact.csv", "complex_noisy.csv", "complex_real.csv"):
        rk = res[f"rank:{name}"]
        rows.append([name, "rank 4 candidates", " > ".join(f["name"] for f in rk["ranking"]),
                     "yes" if rk["best"] == "truth" and rk["decided"] else "NO", f"{rk['seconds']:.0f} s"])
    for start in ("far", "similar", "close"):
        r = res[f"repair:{start}"]
        rows.append(["complex_noisy.csv", f"repair '{start}'", "; ".join(r["best"]["reactions"]),
                     "yes" if same_network(r["best"]["reactions"], ctruth) else "NO", f"{r['seconds']:.0f} s"])
    rows.append([SHORT, f"automatic search ({sc['restarts']} restart{'s' if sc['restarts'] != 1 else ''})",
                 "; ".join(sc["best"]["reactions"]) + f" ({sc['best']['model']})", "yes" if sc_hit else "NO", f"{sc['seconds']:.0f} s"])
    rows.append([SHORT, "rank 4 candidates", " > ".join(f["name"] for f in sr["ranking"]),
                 "yes" if sr["best"] == "truth" and sr["decided"] else ("tied" if sr["best"] == "truth" else "NO"), f"{sr['seconds']:.0f} s"])
    rows.append([ZERO, f"automatic search ({zo['restarts']} restart{'s' if zo['restarts'] != 1 else ''})", "; ".join(zo["best"]["reactions"]),
                 "yes" if zo_hit else ("tied" if zo_tied else "NO"), f"{zo['seconds']:.0f} s"])
    L += md_table(["file", "task", "result", "true network found", "time"], rows)
    L += ["", "Times are for one laptop core and vary by a factor of about two with what else the computer is doing "
          "and with its power mode.", "",
          "### B8. Limits (honest list)", "",
          "- The search can only return reactions that are in its library and rate laws of the power-law form (orders 0, 1, 2), "
          "with at most one adsorption term (product C). A real effect outside that (another species adsorbing, "
          "diffusion limits, a species the GC does not see) shows up as *does not fit within the GC noise*, not as a wrong answer presented as right.",
          "- The search changes one thing at a time (a reaction or one reaction order) and keeps the best change. It is not "
          "guaranteed to find the best network there is. It starts again from another first reaction when its result does "
          "not fit within the GC noise (B5), but a network that fits and is still not the best possible would not be "
          "noticed. Ranking typed candidates (B2) compares exactly the networks you give and has no such limit.",
          "- The search is quickest and most reliable when the true rates follow the equations (mass action), as in the "
          "mystery and the complex example. A true rate with an unusual order (B6) is harder. The search finds it when the "
          "first pass happens to build the right reactions (as on the file of B6), or else from its second new start on, "
          "when it also tries every reaction it may add with one of its orders changed; that took 2 to 3 minutes in the "
          "trials. Those trials were few (seven in all) and of one kind (one reactant of order zero): two unusual orders at "
          "once, or an order other than 0, 1 or 2, may well be missed. A miss is reported as *does not fit within the GC "
          "noise*; the way forward is then to type the suspected network as a candidate and rank it (B2).",
          "- When nothing the search can build fits the data, it starts again up to three times before it gives up, and the "
          "later starts are the slow ones. Such a search can take several minutes (the command line has no time limit; "
          "`discover(..., max_seconds=...)` and `n_restarts` set one).",
          "- *Fits within the GC noise* is judged against the advisor's noise model (1 % of the reading plus 0.05 mol-%). "
          "The noise in the example files is drawn with the two parts independent, which makes the real scatter up to 1.4 "
          "times smaller than that sum; this is why a correct network shows typical misses of 0.5 to 0.8 times the noise "
          "rather than 1, and it makes the test for *fits* lenient by about that factor.",
          "- Two rate laws that predict the same compositions over the runs done cannot be told apart by any method; "
          "the program says so and proposes the run that would decide.",
          "- The run proposed to tell two networks apart is worked out from their fitted numbers. Far from the runs done "
          "those numbers are uncertain, and two fits can then predict very different readings for that reason alone. The "
          "program therefore also gives the difference counted against the GC noise plus that uncertainty, and says so "
          "when one run would not be enough. The uncertainty itself is a rough estimate (a few dozen random draws from each fit).",
          "- All fits use the project's fast bed solver, which agrees with the accurate solver to about 0.1 mol-% of a "
          "reading inside the study window. That can shift a BIC by a few units (about 3 for the true complex network on "
          "`complex_noisy.csv`), one more reason why BIC differences below 6 are not called decisions. Far outside the runs "
          f"done the two differ more: at the run proposed in B1 ({gap['where']}) the readings the two fitted networks "
          f"predict differ between the solvers by up to {gap['mol_pct']:.1f} mol-% ({gap['what']}), so the readings printed "
          "for such a run are approximate (the conclusion that the run separates the networks does not depend on it).",
          "- The 30-run files are designed kinetic studies with a wide range of conditions. The step-limited runs of an "
          f"optimisation campaign cover a much smaller region: on the first 8 runs of {SHORT_TRIALS['n']} such campaigns the "
          f"search found the true network in {SHORT_TRIALS['found']} (B5); in the others it said that its best network does "
          "not fit. Fewer or less varied runs will leave more undecided.",
          "- Equilibrium constants come from the placeholder species table in `smart/discovery.py`; all kinetics are placeholders. "
          "The table was chosen to reproduce the complex network, so the pruning of the library removes exactly the other "
          "forms of its two one-way steps (B1). Without pruning the search returns: on `complex_noisy.csv` "
          f"{UNPRUNED['complex_noisy.csv']}; on `complex_real.csv` {UNPRUNED['complex_real.csv']}; on `{SHORT}` {UNPRUNED[SHORT]}. "
          "With real species the table must come from tabulated thermodynamics.", ""]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-only", action="store_true", help="write only the data and candidate files")
    args = ap.parse_args()
    data = write_data()
    print(f"Wrote {len(data)} data files, {len(D.CANDIDATES)} candidate files and mystery_reactions.txt to {OUT}")
    if args.data_only:
        return
    print("Running the reaction finder on every file (about 10 minutes) ...")
    # the finder is given the files as a user would load them (6 decimals), not the numbers in memory
    res = run_finder({name: D.load_data(os.path.join(OUT, name)) for name in data})
    with open(os.path.join(OUT, "finder_results.json"), "w", encoding="utf-8") as fh:
        json.dump(strip(res), fh, indent=1)
    with open(os.path.join(OUT, "WORKED_EXAMPLE.md"), "w", newline="\n", encoding="utf-8") as fh:
        fh.write("\n".join(worked_example_md(data) + finder_md(res)) + "\n")
    print(f"Wrote WORKED_EXAMPLE.md and finder_results.json to {OUT}")


if __name__ == "__main__":
    main()
