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
    complex_short_campaign.csv   NOT a kinetic study: the first 8 runs of an optimisation campaign on
                            the virtual "real" reactor (the 5 shared screening runs + 3 allowed
                            steps), to show what the finder does with few, step-limited runs
    candidates_truth.txt, candidates_close.txt, candidates_similar.txt, candidates_far.txt
WORKED_EXAMPLE.md           the full calculation of one run by hand, and what the reaction finder
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
from smart.kinetics import GC_NOISE_REL, GC_NOISE_ABS
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
SEEDS = dict(design=7, mystery_noise=5, complex_noise=3, complex_real=11, short_campaign=1)
SHORT = "complex_short_campaign.csv"


def mystery_system():
    return D.build_system(MYSTERY_TEXT, theta=theta_from(MYSTERY_K, MYSTERY_EA), name="mystery")


def complex_system():
    """The complex network with plain rate laws and its true kinetics (same numbers as systems.COMPLEX)."""
    return D.build_system(D.CANDIDATES["truth"], theta=theta_from(COMPLEX_K, COMPLEX_EA), name="complex")


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
    steps towards hotter and higher pressure.  Every run is within MAX_STEP of the one before."""
    X = screening_runs(base_case(0))
    X = np.vstack([X, X[-1] + [5, 0, 0, 0, 20], X[-1] + [10, 5, 0, 0, 40], X[-1] + [10, 10, -5, 0, 60]])
    assert np.all(np.abs(np.diff(X, axis=0)) <= MAX_STEP + 1e-9)
    vr = VirtualReactor(COMPLEX, seed=seed, wcat=WCAT, a_spec=A_SPEC)
    return D.make_data(np.column_stack([X, np.full(len(X), WCAT)]), np.array([vr.run(scale(x)[0])["gc"] for x in X]), name=SHORT)


def datasets():
    """{file name: data} for the six CSV files."""
    phys = study_design()
    ym, yc = exact_gc(mystery_system(), phys), exact_gc(complex_system(), phys)
    return {"mystery_exact.csv": D.make_data(phys, ym, name="mystery_exact.csv"),
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
          "equilibrium moves back to A + B when hotter). Reactions 2 and 3 run one way.", "",
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
          "which is the rounding of the file to 6 decimals (the file itself was made with the project's simulator, "
          "a different integrator from the one used in step 4).", "",
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
          f"One more file, `{SHORT}`, is deliberately different: {len(data[SHORT]['gc'])} step-limited runs from the start of an "
          "optimisation campaign on the virtual \"real\" reactor (see B5).", ""]
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


def finder_md(res):
    def block(lines):
        return ["```"] + lines + ["```", ""]

    my, mx, cr = res["discover:mystery_noisy.csv"], res["discover:mystery_exact.csv"], res["discover:complex_real.csv"]
    L = ["## Part B. What the reaction finder returns", "",
         "How it decides, in short: every network tried is fitted to all GC readings (rate constants and activation "
         "energies). The misfit is counted in units of the GC noise, and networks are compared by BIC = misfit + a "
         "penalty for every fitted number, so an extra reaction must clearly improve the fit to be kept. A BIC "
         "difference below 6 is reported as *the data cannot tell these apart*.", "",
         "### B1. Automatic search on the mystery data", "",
         "    python discover_reactions.py examples/discovery/mystery_noisy.csv", "",
         "The search starts with no reactions and may use any reaction from a library that is **enumerated from the "
         f"atom balance** ({len(D.reaction_library())} reactions among A, B, C, D, E with at most 3 molecules reacting and "
         "2 formed, each one-way in either direction or reversible), not written around the answer.", ""]
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
          "virtual reactor is not in the model at all; its effect is absorbed by the fitted constants.", "",
          "The run that would best tell the first two candidates apart, anywhere in the operating box and within one "
          "allowed step of the last run of the log:", ""]
    L += block(report_next_run(res["next:complex_free"]) + [""] + report_next_run(res["next:complex_step"]))
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
          f"`{SHORT}` is not a kinetic study. It holds the first {sc['best']['n_runs']} runs of an optimisation campaign on the "
          "virtual \"real\" reactor: the 5 shared screening runs from the base case (10 bar, 20 / 20 / 10 mL/min, 600 K: a "
          "temperature ladder, then more pressure, then more B) and 3 more steps to hotter and higher pressure, each within "
          "the allowed step (10 bar, 10 mL/min, 30 K) of the run before. This campaign was chosen because the search needs "
          "more than one pass on it (not every short campaign does): the search changes one reaction at a time, and here "
          "its first choice leads to a dead end. It notices that its result does not fit within the GC noise and starts "
          "again from another first reaction.", ""]
    L += block(report_discovery(sc, SHORT))
    L += [("The search ends at the true reactions" if sc_hit else "The search does NOT end at the true reactions")
          + f" after starting again {sc['restarts']} time{'s' if sc['restarts'] != 1 else ''}. "
          + ("With 8 runs the fit is less sharp than with the 30-run study: compare the uncertainties of the fitted numbers above "
             "with those in B2." if sc_hit else ""), "",
          "Ranking the four typed candidates on the same runs:", ""]
    L += block(report_ranking(sr, SHORT))
    L += ["### B6. Summary", ""]
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
    L += md_table(["file", "task", "result", "true network found", "time"], rows)
    L += ["", "Times are for one laptop core and vary by a factor of about two with what else the computer is doing "
          "and with its power mode.", "",
          "### B7. Limits (honest list)", "",
          "- The search can only return reactions that are in its library and rate laws of the power-law form (orders 0, 1, 2), "
          "with at most one adsorption term (product C). A real effect outside that (another species adsorbing, "
          "diffusion limits, a species the GC does not see) shows up as *does not fit within the GC noise*, not as a wrong answer presented as right.",
          "- The search changes one reaction at a time and keeps the best change. It is not guaranteed to find the best "
          "network there is. It starts again from another first reaction when its result does not fit within the GC noise "
          "(B5), but a network that fits and is still not the best possible would not be noticed. Ranking typed "
          "candidates (B2) compares exactly the networks you give and has no such limit.",
          "- Two rate laws that predict the same compositions over the runs done cannot be told apart by any method; "
          "the program says so and proposes the run that would decide.",
          "- The 30-run files are designed kinetic studies with a wide range of conditions. The step-limited runs of an "
          "optimisation campaign cover a much smaller region; B5 shows one such case that works, but fewer or less varied "
          "runs will leave more undecided.",
          "- Equilibrium constants come from the placeholder species table in `smart/discovery.py`; all kinetics are placeholders.", ""]
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
