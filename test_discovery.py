"""Checks for reaction discovery (smart/discovery.py, discover_reactions.py, examples/discovery/).
Run:  python test_discovery.py              about 6.5 min on a laptop (it runs six full reaction searches,
                                          25 to 90 s each); longer when the laptop is busy
      python test_discovery.py --quick      skips the searches: 40 s to 1.5 min
      python test_discovery.py --thorough   adds one hard search (a zero-order rate on a run design that
                                          needs two new starts): about 2.5 min more"""
import io
import json
import os
import subprocess
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from smart import discovery as D
from smart.systems import COMPLEX, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, unscale, screening_runs, base_case
import make_discovery_example as M
import discover_reactions as CLI

QUICK = "--quick" in sys.argv
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "examples", "discovery")


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def raises(text, *words):
    """True if parsing `text` fails with a ReactionError whose message holds every word."""
    try:
        D.parse_network(text)
    except D.ReactionError as e:
        return all(w in str(e) for w in words)
    return False


def same(a, b):
    return D.network_key(D.parse_network(a)) == D.network_key(D.parse_network(b))


# ----------------------------------------------------------------------------------- 1. text format
texts = ["A + B <=> C", "C + B -> E", "A -> D", "A -> D ; orders A=2", "C + B -> E ; orders C=1, B=0",
         "A + 2 B -> E ; orders A=1, B=1", "2 C -> A + E", "A + B <=> C ; orders A=2, B=1"]
check("parser round trip: text -> reaction -> text -> the same reaction",
      all(D.parse_reaction(D.parse_reaction(t).text()) == D.parse_reaction(t) and D.parse_reaction(t).text() == t for t in texts))
check("the order in which species are written does not matter (B + A -> C is A + B -> C; 2B is 2 B)",
      D.parse_reaction("B + A -> C") == D.parse_reaction("A + B -> C") and D.parse_reaction("A+2B->E") == D.parse_reaction("A + 2 B -> E")
      and D.parse_reaction("A + B <-> C") == D.parse_reaction("A + B <=> C") and D.parse_reaction("A + B --> C") == D.parse_reaction("A + B -> C"))
net = D.parse_network("# a comment\nA + B <=> C\n\nC + B -> E   # over-reaction\nA -> D\n")
check("a network file: comments and blank lines are ignored; network_text round-trips",
      len(net) == 3 and D.parse_network(D.network_text(net)) == net)
r = D.parse_reaction("C + B -> E ; orders C=1, B=0")
check("orders: default = coefficients (mass action); 'orders' overrides; rate laws are written out",
      D.parse_reaction("A + 2 B -> E").orders == {"A": 1.0, "B": 2.0} and r.orders == {"C": 1.0, "B": 0.0}
      and D.parse_reaction("A + 2 B -> E").rate_law(3) == "r3 = k3 P_A P_B^2" and "zero order in B" in r.rate_law()
      and D.parse_reaction("A + B <=> C").rate_law(1) == "r1 = k1 (P_A P_B - P_C / K1)")
check("a reversible reaction with non-default orders keeps a rate that is zero at equilibrium",
      D.parse_reaction("A + B <=> C ; orders A=2, B=1").rate_law() == "r = k (P_A^2 P_B - P_A P_C / K)")
check("plain-word errors: no arrow, unknown species, unbalanced atoms, the inert, same species on both sides",
      raises("A + B C", "arrow") and raises("A + X -> C", "Unknown species 'X'") and raises("A + B -> E", "not balanced")
      and raises("A + I -> C", "inert") and raises("A + C -> C + D", "both sides"))
check("plain-word errors: bad order, order below the coefficient in a reversible reaction, a reaction listed twice",
      raises("A -> D ; orders A=7", "between 0 and 3") and raises("A + B <=> C ; orders B=0", "reverse rate")
      and raises("A + B -> C\nC -> A + B", "listed twice") and raises("A -> D ; ordrs A=2", "orders"))
check("plain-word errors: nothing on one side of the arrow, two arrows, a fractional or zero coefficient, a coefficient too "
      "large to serve as the order",
      raises("A -> ", "nothing is written after the arrow") and raises("-> C", "nothing is written before the arrow")
      and raises("A -> D -> E", "Could not read 'D -> E'") and raises("1.5 A -> 1.5 D", "whole-number")
      and raises("0 A + B -> C", "1 or more") and raises("4 A -> 4 D", "taken from the coefficient"))
check("atom balance: C = AB, D = one A unit, E = AB2",
      D.atom_count({"C": 1, "B": 1}) == D.atom_count({"E": 1}) == {"A": 1, "B": 2} and D.atom_count({"D": 2}) == {"A": 2})

# ----------------------------------------------------------------------------------- 2. the species table and the library
t1 = D.parse_reaction("A + B <=> C").thermo()
check("species table reproduces COMPLEX.thermo (A + B <=> C) and COMPLEX.dH (all three reactions)",
      np.allclose(t1, COMPLEX.thermo[0]) and np.allclose([r_.thermo()[0] for r_ in D.parse_network(D.CANDIDATES["truth"])], COMPLEX.dH))
lib = D.reaction_library()
variants = [D.parse_reaction(v["text"]) for e in lib for v in e["variants"]]      # parsing checks the atom balance
nus = []
for e in lib:
    S_ = D.build_system([e["variants"][0]["text"]])
    nus.append(tuple(np.round(S_.nu[:, 0] * np.sign(S_.nu[np.nonzero(S_.nu[:, 0])[0][0], 0])).astype(int)))
check("library: every reaction is atom-balanced, no entry twice (also not as its reverse or a multiple)",
      len({e["key"] for e in lib}) == len(lib) == len(set(nus)) and len({v.key() for v in variants}) == len(variants))
check("library: at most 3 molecules react and 2 are formed in every one-way variant; no species on both sides",
      all(sum(v.reactants.values()) <= 3 and sum(v.products.values()) <= 2 for v in variants if not v.reversible)
      and all(not set(v.reactants) & set(v.products) for v in variants))
check("library: 12 balanced reactions among A, B, C, D, E (the number found by hand enumeration)", len(lib) == 12)
full = [D.parse_reaction(v["text"]) for e in D.reaction_library(thermo_prune=False) for v in e["variants"]]
check("library: without the thermodynamic pruning every entry has its reversible variant; with it, fewer variants",
      sum(v.reversible for v in full) == 12 and len(variants) < len(full))
keys = {v.key() for v in variants}
check("library (generated by enumeration) contains the reactions of both examples",
      all(D.parse_reaction(t).key() in keys for t in ("A + B <=> C", "C + B -> E", "A -> D", "C -> D + B", "A + 2 B -> E")))

# ----------------------------------------------------------------------------------- 3. networks as reactor models
S = D.candidate_system("truth")
X = np.random.default_rng(0).random((40, 5))
P, VA, VB, VI, T = unscale(X).T
F1, F0s = S.simulate(P, VA, VB, VI, 1.0, T, theta=COMPLEX.theta_true)
F2, _ = COMPLEX.simulate(P, VA, VB, VI, 1.0, T, theta=COMPLEX.theta_true)
Fx, _ = S.simulate(P, VA, VB, VI, 1.0, T, theta=COMPLEX.theta_true, method="lsoda")
Fy, _ = COMPLEX.simulate(P, VA, VB, VI, 1.0, T, theta=COMPLEX.theta_true, method="lsoda")
FT = F0s.sum(1, keepdims=True)
print(f"     truth network against the built-in complex system: fast solvers differ by {np.max(np.abs(F1 - F2) / FT):.1e} of the feed")
check("build_system('truth') is the built-in complex system (same stoichiometry, orders, thermo; the accurate solver gives "
      "identical outlet flows, the two fast solvers agree to 1e-6 of the feed)",
      np.array_equal(S.nu, COMPLEX.nu) and np.array_equal(S.fwd, COMPLEX.fwd) and np.array_equal(S.rev, COMPLEX.rev)
      and S.thermo == COMPLEX.thermo and np.array_equal(Fx, Fy) and np.max(np.abs(F1 - F2) / FT) < 1e-6)
check("generic starting guess has the magnitude of the built-in guesses", np.allclose(S.theta_guess, COMPLEX.theta_guess))


def solver_miss(S_, theta, inhib=None):
    """Largest difference between the networks' fast integrator and LSODA, as a share of the feed."""
    Fa, _ = S_.simulate(P, VA, VB, VI, 1.0, T, theta=theta, inhib=inhib)
    Fb, _ = S_.simulate(P, VA, VB, VI, 1.0, T, theta=theta, inhib=inhib, method="lsoda")
    return float(np.max(np.abs(Fa - Fb) / FT))


Sc, Sf = D.candidate_system("close"), D.candidate_system("far")
errs = [solver_miss(S, COMPLEX.theta_true), solver_miss(S, COMPLEX.theta_true, inhib=0.03),
        solver_miss(Sc, Sc.theta_guess + np.tile([2.0, 0.0], 3)), solver_miss(Sf, Sf.theta_guess + np.tile([2.0, 0.0], 2)),
        solver_miss(Sf, Sf.theta_guess + np.tile([2.0, 0.0], 2), inhib=0.1)]
print("     fast integrator against LSODA, share of the feed:", ", ".join(f"{e:.1e}" for e in errs))
check("fast integrator of the networks within 2e-3 of LSODA over the whole box (plain, site blocking, zero-order reactants)",
      max(errs) < 2e-3)
dn = D.load_data(os.path.join(EX, "complex_noisy.csv"))
nets_ = [D.parse_network(D.CANDIDATES[n]) for n in D.CANDIDATES] + [D.parse_network("A -> D")]
ths_ = [D.generic_theta(len(n_)) + 0.3 * np.random.default_rng(i_).standard_normal(2 * len(n_)) for i_, n_ in enumerate(nets_)]
sig_ = D.GC_REL * dn["gc"][:, :-1] + D.GC_ABS
worst_r = worst_J = 0.0
for n_, th_, st_ in zip(nets_, ths_, D._batch_lm(nets_, dn, ths_, n_iter=0)):
    kf_ = D.KineticFit(D.build_system(n_), models=("plain",))
    r_, J_ = kf_._res_jac(th_, dn["phys"], dn["gc"], sig_, kf_.time_on_stream(dn["hours"]), kf_.guess("plain"), kf_.prior_sd("plain"), "plain")
    worst_r, worst_J = max(worst_r, np.abs(r_ - st_["r"]).max()), max(worst_J, np.abs(J_ - st_["J"]).max())
print(f"     networks fitted together against one at a time: residuals differ by {worst_r:.1e}, derivatives by {worst_J:.1e}")
check("the search's batch (different networks integrated together) gives each network the residuals and derivatives "
      "of the ordinary kinetic fit", worst_r < 1e-9 and worst_J < 1e-6)
Ff, F0f = Sf.simulate(P, VA, VB, VI, 1.0, T, theta=Sf.theta_guess + np.tile([4.0, 0.0], 2), method="lsoda")
check("zero-order reactant: the reaction stops when that reactant runs out (no negative flow, atoms conserved)",
      np.all(np.isfinite(Ff)) and Ff.min() >= 0 and np.allclose(Ff[:, 0] + Ff[:, 2] + Ff[:, 3], F0f[:, 0], rtol=1e-6)
      and (Ff[:, 1] / F0f[:, 1]).min() < 0.02)
check("the four example candidates are built from their text; 'far' has 2 reactions and makes no E",
      [D.candidate_system(n).nr for n in D.CANDIDATES] == [3, 3, 3, 2] and not D.candidate_system("far").nu[4].any())

# ----------------------------------------------------------------------------------- 4. the example files reproduce
data = {n: D.load_data(os.path.join(EX, n)) for n in ("mystery_exact.csv", "mystery_noisy.csv", "complex_exact.csv",
                                                       "complex_noisy.csv", "complex_real.csv")}
fresh = M.datasets()
check("example files hold the website's columns and 30 runs; temperatures are read as kelvin",
      all(len(d["gc"]) == 30 and d["gc"].shape[1] == 6 for d in data.values())
      and open(os.path.join(EX, "complex_noisy.csv")).readline().strip().split(",")[1:7] == D.LOG_COLS
      and 500 < data["complex_noisy.csv"]["phys"][:, 4].min() and np.allclose(data["complex_noisy.csv"]["gc"].sum(1), 1))
ph = data["mystery_exact.csv"]["phys"]
ym = M.exact_gc(M.mystery_system(), ph)
yc = M.exact_gc(M.complex_system(), ph)
check("mystery_exact.csv = the stated reactions and rate laws re-simulated from the set points in the file",
      np.max(np.abs(ym - data["mystery_exact.csv"]["gc"])) < 2e-6)
check("complex_exact.csv = the complex network re-simulated (and equals the built-in COMPLEX kinetics)",
      np.max(np.abs(yc - data["complex_exact.csv"]["gc"])) < 2e-6 and np.allclose(M.complex_system().theta_true, COMPLEX.theta_true))
check("the noisy and 'real' files reproduce from their seeds; the candidate files hold the candidate texts",
      all(np.max(np.abs(fresh[n]["gc"] - data[n]["gc"])) < 2e-6 and np.allclose(fresh[n]["phys"], data[n]["phys"]) for n in data)
      and all(same(open(os.path.join(EX, f"candidates_{n}.txt")).read(), D.CANDIDATES[n]) for n in D.CANDIDATES))
i = M.worked_run(M.mystery_system(), data["mystery_exact.csv"])
prof = M.bed_profile(M.mystery_system(), ph[i], np.array([0.0, 50.0, 100.0]))
check("the hand calculation of WORKED_EXAMPLE.md (direct integration along the bed) matches its row of the file",
      np.max(np.abs(prof[-1] / prof[-1].sum() - data["mystery_exact.csv"]["gc"][i])) < 2e-6)


def by_hand(name, row):
    """GC mole fractions of one run of an exact file, from the rate laws WRITTEN OUT HERE (the
    numbers of WORKED_EXAMPLE.md, A2 and A4) and scipy's Radau integrator: nothing of the
    project's simulator, systems.py or discovery.py is used."""
    from scipy.integrate import solve_ivp
    P, VA, VB, VI, T, W = data[name]["phys"][row]
    mol = lambda v: 101325.0 * (v * 1e-6 / 60.0) / (8.314 * 273.15) * 1e6          # mL/min (standard) -> micromol/s
    k = lambda k300, Ea: k300 / 100.0 * np.exp(-Ea / 8.314 * (1 / T - 1 / 573.15))  # per m2 at T (100 m2/g)
    K1 = np.exp(120e3 / (8.314 * T) - 170.0 / 8.314)

    def rhs(_, F):
        A, B, C, D_, E, I = np.maximum(F, 0.0)
        pA, pB, pC = (P * v / (A + B + C + D_ + E + I) for v in (A, B, C))
        if name.startswith("mystery"):           # A + B <=> C;  C -> D + B;  A + 2 B -> E
            r1, r2, r3 = k(2e-6, 65e3) * (pA * pB - pC / K1), k(1e-6, 110e3) * pC, k(6e-8, 85e3) * pA * pB ** 2
            return np.array([-r1 - r3, -r1 + r2 - 2 * r3, r1 - r2, r2, r3, 0.0]) * 1e6
        r1, r2, r3 = k(3e-6, 70e3) * (pA * pB - pC / K1), k(3e-7, 120e3) * pC * pB, k(1e-6, 150e3) * pA   # A + B <=> C;  C + B -> E;  A -> D
        return np.array([-r1 - r3, -r1 - r2, r1 - r2, r3, r2, 0.0]) * 1e6

    F = solve_ivp(rhs, (0.0, 100.0 * W), [mol(VA), mol(VB), 0, 0, 0, mol(VI)], method="Radau", rtol=1e-11, atol=1e-13).y[:, -1]
    return F / F.sum()


hand = max(np.max(np.abs(by_hand(n, row) - data[n]["gc"][row])) for n in ("mystery_exact.csv", "complex_exact.csv") for row in (0, 9, 18, 29))
print(f"     rate laws written out by hand against the exact files (4 runs of each): largest difference {hand:.1e}")
check("mystery_exact.csv and complex_exact.csv equal an independent calculation from the rate laws written out by hand "
      "(no project simulator), to the 6 decimals of the files", hand < 1e-6)
noise_z = (data["complex_noisy.csv"]["gc"] - data["complex_exact.csv"]["gc"]) / D.gc_noise(data["complex_exact.csv"]["gc"])
check("the noise in the noisy file is of the size of the GC noise model (typical 0.5 to 1.1 of it)",
      0.5 < np.sqrt(np.mean(noise_z[:, :-1] ** 2)) < 1.1)
buf = io.StringIO(D.data_to_csv(data["complex_noisy.csv"]))
no_run = "\n".join(ln.split(",", 1)[1] for ln in buf.getvalue().splitlines()) + "\n"      # the same log without the 'run' column
check("data_to_csv / load_data round trip (from text, from an open file, and from uploaded text without the 'run' column "
      "that starts with a byte-order mark)",
      np.allclose(D.load_data(buf.getvalue())["gc"], data["complex_noisy.csv"]["gc"], atol=1e-6)
      and np.allclose(D.load_data("\ufeff" + no_run)["gc"], data["complex_noisy.csv"]["gc"], atol=1e-6)
      and np.allclose(D.load_data(buf)["phys"], data["complex_noisy.csv"]["phys"]))
check("detected products: C, D, E in the complex data", list(D.detected_species(data["complex_noisy.csv"])) == ["C", "D", "E"])


def load_fails(text, *words):
    """True if load_data refuses the CSV text with a ValueError whose message holds every word."""
    try:
        D.load_data(text)
    except ValueError as e:
        return all(w in str(e) for w in words)
    return False


head_ = "pressure_bar,flow_A_mL_min,flow_B_mL_min,flow_I_mL_min,temperature_C,catalyst_g,gc_A,gc_B,gc_C,gc_D,gc_E,gc_I\n"
check("a bad run log is refused in plain words that name the run and the column (not a number, empty, negative, no GC "
      "reading, missing columns, semicolons)",
      load_fails(head_ + "10,20,20,10,300,1,0.2,0.3,0.2,0.1,0.1,0.1\n10,20,20,10,300,1,abc,0.3,0.2,0.1,0.1,0.1\n", "Run 2", "gc_A", "not a number")
      and load_fails(head_ + "10,20,20,10,,1,0.2,0.3,0.2,0.1,0.1,0.1\n", "Run 1", "temperature_C", "empty")
      and load_fails(head_ + "10,20,20,10,300,1,-0.2,0.3,0.2,0.1,0.1,0.1\n", "Run 1", "gc_A")
      and load_fails(head_ + "10,20,20,10,300,1,0,0,0,0,0,0\n", "Run 1", "no GC reading")
      and load_fails("pressure_bar,flow_A_mL_min\n1,2\n", "missing the columns", "gc_A")
      and load_fails(head_.replace(",", ";") + "10;20;20;10;300;1;0.2;0.3;0.2;0.1;0.1;0.1\n", "separated by commas"))
row_ = "10,20,20,10,300,1,0.2,0.3,0.2,0.1,0.1,0.1\n"
spaced = D.load_data(head_.replace(",", ", ") + row_.replace(",", ", "))
utf16 = D.load_data((head_ + row_).encode("utf-16"))
check("load_data reads a header with spaces after the commas and a UTF-16 file (with its byte-order mark) like the plain file",
      np.allclose(spaced["gc"], D.load_data(head_ + row_)["gc"]) and np.allclose(utf16["phys"], spaced["phys"]))
check("load_data in plain words: an empty source, a file that does not exist, text in an encoding it cannot read, and an "
      "EMPTY GC cell (a reading left out is not a reading of zero: the run and the column are named)",
      load_fails("", "empty") and load_fails("not a csv", "Could not find the file") and load_fails("missing_log.csv", "Could not find")
      and load_fails(b"\xe9\xe8" + (head_ + row_).encode("latin-1"), "UTF-8", "CSV UTF-8")
      and load_fails(head_ + row_ + "10,20,20,10,300,1,0.2,0.3,0.2,0.1,,0.1\n", "Run 2", "gc_E", "empty", "write 0"))
noE_ = D.load_data(head_.replace(",gc_E", "") + "10,20,20,10,300,1,0.2,0.3,0.2,0.1,0.2\n")
hotK = D.load_data(head_ + "10,20,20,10,650,1,0.2,0.3,0.2,0.1,0.1,0.1\n")
check("a log without a gc_E column is read with E = 0 and a note that says so; a temperature above the box (typed in "
      "kelvin?) gets a note; both notes are repeated in the summaries",
      noE_["gc"][0, 4] == 0 and any("no gc_E column" in n_ for n_ in noE_["notes"])
      and any("kelvin" in n_ for n_ in hotK["notes"]) and not D.load_data(head_ + row_)["notes"]
      and any("no gc_E column" in s_ for s_ in D.rank_candidates({"t": "A + B <=> C"}, noE_, models=("plain",), n_starts=1)["summary"]))

# ----------------------------------------------------------------------------------- 5. ranking the candidates
ranks = {}
for name, ns in (("complex_exact.csv", 1), ("complex_noisy.csv", 2)):
    t = time.time()
    ranks[name] = rk = D.rank_candidates(D.CANDIDATES, data[name], n_starts=ns)
    order = [f["name"] for f in rk["ranking"]]
    by = {f["name"]: f for f in rk["ranking"]}
    print(f"     {name}: " + ", ".join(f"{n} BIC {by[n]['bic']:.0f}" for n in order) + f"   ({time.time() - t:.0f} s)")
    check(f"{name}: ranking is truth, close, similar, far", order == ["truth", "close", "similar", "far"])
    check(f"{name}: the truth fits within the GC noise and is not tied with any other candidate",
          by["truth"]["fits_noise"] and rk["decided"] and rk["best"] == "truth" and by["close"]["delta_bic"] > D.INDISTINGUISHABLE)
    check(f"{name}: 'far' is reported as unable to explain E (nothing makes it); 'close' and 'similar' miss by more than the noise",
          by["far"]["not_made"] == ["E"] and "cannot explain E" in by["far"]["verdict"] and by["close"]["unexplained"]
          and by["similar"]["unexplained"] and not by["close"]["fits_noise"])
f = ranks["complex_noisy.csv"]["ranking"][0]
k = np.array([r_["k_ref"] for r_ in f["rate_constants"]])
Ea = np.array([r_["Ea_kJ"] for r_ in f["rate_constants"]])
sd = np.array([r_["Ea_sd_kJ"] for r_ in f["rate_constants"]])
print("     fitted k:", k, " Ea:", Ea.round(1), "+-", sd.round(1))
check("complex_noisy.csv: fitted rate constants within 10 % and activation energies within 3 sd (and 5 kJ/mol) of the truth",
      np.all(np.abs(k / np.array(M.COMPLEX_K) - 1) < 0.10) and np.all(np.abs(Ea - np.array(M.COMPLEX_EA) / 1000) < np.minimum(3 * sd + 0.5, 5.0)))
check("fit report: chi2, BIC, parameters and misses per species are consistent",
      f["n_params"] == 6 and f["n_data"] == 150 and abs(f["bic"] - (f["chi2"] + 6 * np.log(150))) < 1e-9
      and set(f["miss"]) == {"A", "B", "C", "D", "E"} and all(m["x_noise"] < 1.2 for m in f["miss"].values()))
check("results are JSON-friendly", len(json.dumps(ranks["complex_noisy.csv"])) > 100)
two = D.rank_candidates({"one way": "A + B <=> C\nC + B -> E\nA -> D", "D both ways": "A + B <=> C\nC + B -> E\nA <=> D"},
                        data["complex_noisy.csv"], models=("plain",), n_starts=1)
print("     tie test: BIC difference", round(two["ranking"][1]["delta_bic"], 3), "|", two["undecided"])
check("two networks the data cannot tell apart (A -> D one way or both ways) are reported as undecided, not ranked",
      not two["decided"] and len(two["tied"]) == 1 and "cannot decide" in two["undecided"][0])
bad_rk = D.rank_candidates({"far": D.CANDIDATES["far"], "no E": "A + B -> C\nA -> D"}, data["complex_noisy.csv"], models=("plain",), n_starts=1)
bad_txt = "\n".join(CLI.report_ranking(bad_rk))
check("when no candidate fits, the report says so and calls the first one the least bad, never 'the data support' it",
      not bad_rk["ranking"][0]["fits_noise"] and "The data support" not in bad_txt and "None of the candidates fits" in bad_txt
      and "least bad" in bad_rk["summary"][0] and "The data support 'truth'." in "\n".join(CLI.report_ranking(ranks["complex_noisy.csv"])))
try:
    D.rank_candidates({}, data["complex_noisy.csv"])
    empty_ok = False
except ValueError as e:
    empty_ok = "at least one" in str(e)
dn_ = data["complex_noisy.csv"]
feed_only = D.make_data(dn_["phys"], np.column_stack([dn_["phys"][:, 1], dn_["phys"][:, 2], np.zeros((30, 3)), dn_["phys"][:, 3]]))
nothing = D.discover(feed_only)
one_run = D.rank_candidates({"truth": D.CANDIDATES["truth"]}, D.make_data(dn_["phys"][:1], dn_["gc"][:1]), models=("plain",), n_starts=1)
try:
    D.fit_network(D.CANDIDATES["truth"], dn_, models=("plain", "decay"))
    models_ok = False
except ValueError as e:
    models_ok = "'decay'" in str(e) and "'plain'" in str(e) and "'inhib'" in str(e)
dup6 = D.make_data(np.repeat(dn_["phys"][7:8], 6, 0), np.repeat(dn_["gc"][7:8], 6, 0))
dup_rk = D.rank_candidates({"truth": D.CANDIDATES["truth"]}, dup6, models=("plain",), n_starts=1)
check("six repeats of ONE setting count as one setting: the result carries the few-runs caution, which says so (a 30-run "
      "study of 30 settings carries no caution)",
      D.n_settings(dup6) == 1 and dup_rk["n_settings"] == 1 and "only 1 different setting" in dup_rk["summary"][-1]
      and D.n_settings(dn_) == 30 and not D._cautions(dn_, 6))
from smart.systems import SYSTEMS, SIMPLE
from smart.advisor import start_campaign
_, _, hsim, _ = start_campaign("simple", 0)
dsim = D.make_data(np.column_stack([unscale(np.array(hsim["X"])), np.ones(5)]), np.array(hsim["gc"]))
fs_ = D.fit_network(SYSTEMS["simple"], dsim, n_starts=1)
rs_ = D.rank_candidates({"current model": SIMPLE, "A only": "A -> D"}, dsim, n_starts=1)
check(f"the built-in simple system (5 GC columns, no E) can be fitted and ranked as a ReactionSystem object, not only as "
      f"text (typical miss {fs_['x_noise']:.2f} times the GC noise)",
      fs_["n_reactions"] == 2 and np.isfinite(fs_["x_noise"]) and fs_["x_noise"] < 2 and rs_["best"] == "current model"
      and abs(fs_["bic"] - D.fit_network(D.network_text(SIMPLE), dsim, n_starts=1)["bic"]) < 1e-6)
check("edge cases in plain words: no candidates given; a log in which nothing reacts; a log with a single run carries a "
      "caution; a model class that discovery does not use is refused",
      empty_ok and nothing["best"]["reactions"] == [] and "No reaction is needed" in nothing["summary"][0]
      and list(D.detected_species(D.make_data(dn_["phys"][:1], dn_["gc"][:1]))) == ["C", "D", "E"] and "only 1 run" in one_run["summary"][-1]
      and models_ok)

# ----------------------------------------------------------------------------------- 6. the run that tells two networks apart
d = data["complex_noisy.csv"]
nr = D.discriminating_run(ranks["complex_noisy.csv"], d)
x = np.array(nr["x"])
print("     " + nr["message"])
check("discriminating run (free): inside the operating box, enough flow for the GC, catalyst mass unchanged",
      np.all(x >= LOW - 1e-9) and np.all(x <= HIGH + 1e-9) and x[1:4].sum() >= MIN_TOTAL_FLOW and nr["run"]["catalyst_g"] == d["phys"][-1, 5])
check("discriminating run: truth and close differ there by far more than the GC noise, and more than in any run done",
      nr["networks"] == ["truth", "close"] and nr["separation"] > 10 and nr["separation"] > nr["best_so_far"])
check("discriminating run: the difference is also reported allowing for the uncertainty of both fits (never larger than "
      "the best-fit difference), and the expected gain in BIC and the message are based on that number",
      0 < nr["separation_uncertain"] <= nr["separation"] + 1e-9 and abs(nr["expected_bic_gap"] - nr["separation_uncertain"] ** 2) < 1e-6
      and "uncertain" in nr["message"] and nr["separation_uncertain"] > 10)
x_last = d["phys"][-1, :5]
nr2 = D.discriminating_run(ranks["complex_noisy.csv"], d, x_last=x_last)
x2 = np.array(nr2["x"])
check("discriminating run (step-limited): within one allowed step of the last run and inside the box",
      np.all(np.abs(x2 - x_last) <= MAX_STEP + 1e-9) and np.all(x2 >= LOW - 1e-9) and np.all(x2 <= HIGH + 1e-9)
      and x2[1:4].sum() >= MIN_TOTAL_FLOW and nr2["step_limited"] and nr2["separation"] <= nr["separation"] + 1e-9)
check("discriminating run: the message says whether the run is a free one or within one allowed step of the last run",
      "a free run" in nr["message"] and not nr["step_limited"] and "within one allowed step of the last run" in nr2["message"]
      and "a free run" not in nr2["message"] and "free run" in "\n".join(CLI.report_next_run(nr))
      and "free run" not in "\n".join(CLI.report_next_run(nr2)))
x3 = np.array(D.discriminating_run(ranks["complex_noisy.csv"], d, x_last=HIGH + [40.0, 10.0, 10.0, 10.0, 50.0])["x"])
check("discriminating run: a last run OUTSIDE the box still gives a run inside the box",
      np.all(x3 >= LOW - 1e-9) and np.all(x3 <= HIGH + 1e-9) and x3[1:4].sum() >= MIN_TOTAL_FLOW)
nr4 = D.discriminating_run(CLI.strip(json.loads(json.dumps(ranks["complex_noisy.csv"]))), d)
x4 = np.array(nr4["x"])
print(f"     from a result saved without its covariance: {nr4['separation']:.0f}x the GC noise apart (with it: {nr['separation']:.0f}x)")
# without the covariance the run is chosen on the best-fit predictions alone, so it need not be the
# same point; it must still be a valid run that separates the two networks at least as well
check("discriminating run also works on a result saved without its bulky parts (discover_reactions.strip)",
      np.all(x4 >= LOW - 1e-9) and np.all(x4 <= HIGH + 1e-9) and x4[1:4].sum() >= MIN_TOTAL_FLOW
      and nr4["networks"] == ["truth", "close"] and nr4["separation"] >= nr["separation"] - 1e-6
      and abs(nr4["separation_uncertain"] - nr4["separation"]) < 1e-6)
yp = D.predict_gc(ranks["complex_noisy.csv"]["ranking"][0], np.r_[x, 1.0][None, :])[0]
check("predict_gc of a fitted network agrees with the prediction reported for the run; JSON-friendly",
      np.allclose(100 * yp, list(nr["predicted"]["truth"].values()), atol=1e-6) and len(json.dumps(nr)) > 50)

# ----------------------------------------------------------------------------------- 6b. one step of the search, and the yardstick, on their own
# (what the advisor uses to check and repair its reaction network during a campaign: SmartAdvisor(revise=True))
f_truth, f_far = ranks["complex_noisy.csv"]["ranking"][0], ranks["complex_noisy.csv"]["ranking"][-1]
m_truth = D.misfit(D.CANDIDATES["truth"], d, np.array(f_truth["predicted_gc"]), f_truth["n_params"])
m_far = D.misfit(D.CANDIDATES["far"], d, np.array(f_far["predicted_gc"]), f_far["n_params"])
print(f"     misfit of the fitted truth: {m_truth['x_noise']:.2f}x the GC noise, {m_truth['x_allow']:.2f}x the allowance for model error; "
      f"of 'far': {m_far['x_noise']:.1f}x and {m_far['x_allow']:.1f}x")
check("misfit(): the yardstick on its own gives the numbers of the fit report (typical miss, species not made, fits or not)",
      abs(m_truth["x_noise"] - f_truth["x_noise"]) < 1e-3 and m_truth["fits"] and m_truth["within_allowance"] and not m_truth["beyond"]
      and m_far["not_made"] == ["E"] and not m_far["fits"] and "cannot explain E" in m_far["verdict"]
      and abs(m_far["x_noise"] - f_far["x_noise"]) < 1e-2 * f_far["x_noise"])
check("misfit(): the allowance for model error (2 % of the reading + 0.2 mol-%) is the more forgiving yardstick, and a "
      "network that makes no E is outside it as well",
      m_truth["x_allow"] < m_truth["x_noise"] and m_far["x_allow"] < m_far["x_noise"] and not m_far["within_allowance"]
      and m_far["x_allow"] > 1 and len(json.dumps(m_far)) > 50)
d_bad = dict(d, gc=d["gc"].copy())
d_bad["gc"][7, 3] = 0.0                                    # one wrong reading: D of run 8 typed as 0
d_bad["gc"][7] /= d_bad["gc"][7].sum()
m_bad = D.misfit(D.CANDIDATES["truth"], d_bad, np.array(f_truth["predicted_gc"]), f_truth["n_params"])
print("     one wrong reading: " + m_bad["verdict"])
check("misfit(): when one run carries the misfit (a wrong reading), it says which run and that its GC reading should be "
      "checked; the other runs are explained within the allowance",
      not m_bad["fits"] and m_bad["one_run"] and m_bad["worst_run"] == 8 and m_bad["rest_within_allowance"]
      and m_bad["worst_reading"]["species"] == "D" and "run 8 alone accounts for" in m_bad["verdict"]
      and not m_truth["one_run"] and "alone accounts" not in m_truth["verdict"])
# site blocking at a GIVEN strength in the batch fit (what one_step_revision screens with when the caller's fit found it)
# The reference is the network's own simulator with the same inhibition, integrating the same rows the same way (each
# run with its perturbed copies, which share step sizes), as the kinetic fit does; the kinetic fit's 'inhib' model
# itself perturbs ln K_C as well, so its step sizes, and its numbers at the 1e-7 level, differ.
S_t = D.build_system(D.CANDIDATES["truth"])
th_t = S_t.theta_guess + 0.3
K_t = np.exp(np.log(0.03) + D.PRIOR_E * (D.T_REF / d["phys"][:, 4] - 1.0))
n_r, p_t, eps_t = len(d["gc"]), len(th_t), 1e-4
TH_t = np.repeat(th_t[None], p_t + 1, 0)
TH_t[1:] += eps_t * np.eye(p_t)
ph_t = np.tile(d["phys"], (p_t + 1, 1))
F_t, _ = S_t.simulate(ph_t[:, 0], ph_t[:, 1], ph_t[:, 2], ph_t[:, 3], ph_t[:, 5], ph_t[:, 4], theta=np.repeat(TH_t, n_r, 0),
                      groups=np.tile(np.arange(n_r), p_t + 1), a_spec=d["a_spec"], inhib=np.tile(K_t, p_t + 1))
Y_t = (F_t / F_t.sum(1, keepdims=True)).reshape(p_t + 1, n_r, -1)[..., :-1]
sig_t = D.GC_REL * d["gc"][:, :-1] + D.GC_ABS
r_t = ((Y_t[0] - d["gc"][:, :-1]) / sig_t).ravel()
J_t = ((Y_t[1:] - Y_t[0][None]) / eps_t / sig_t[None]).reshape(p_t, -1).T
st_t = D._batch_lm([D.parse_network(D.CANDIDATES["truth"])], d, [th_t], n_iter=0, inhib=[K_t])[0]
st_0 = D._batch_lm([D.parse_network(D.CANDIDATES["truth"])], d, [th_t], n_iter=0)[0]
check("the batch fit with site blocking at a given strength gives the residuals and derivatives of the network's own "
      "simulator with that site blocking (and differs from plain rate laws)",
      np.abs(r_t - st_t["r"][:r_t.size]).max() < 1e-9 and np.abs(J_t - st_t["J"][:r_t.size]).max() < 1e-6
      and np.abs(r_t - st_0["r"][:r_t.size]).max() > 0.1)
no_E = "A + B <=> C\nA -> D"                               # the complex network with the over-reaction left out
t = time.time()
one = {n_: D.one_step_revision(net_, d) for n_, net_ in (("no E", no_E), ("close", D.CANDIDATES["close"]), ("truth", D.CANDIDATES["truth"]))}
for n_, r_ in one.items():
    b_ = r_["moves"][0]
    print(f"     one step from '{n_}' ({r_['n_tried']} single changes tried): best is to {b_['what']}; BIC {r_['current']['bic']:.0f} -> {b_['fit']['bic']:.0f}")
print(f"     ({time.time() - t:.0f} s for the three)")
check("one step from the network without C + B -> E: the best single change adds exactly that reaction, and E is then explained",
      one["no E"]["moves"][0]["kind"] == "add" and same(one["no E"]["moves"][0]["reactions"], D.CANDIDATES["truth"])
      and one["no E"]["current"]["not_made"] == ["E"] and one["no E"]["moves"][0]["fit"]["fits_noise"] and one["no E"]["moves"][0]["gain"] > 1000)
check("one step from 'close' (one order wrong): the best single change puts the order right",
      one["close"]["moves"][0]["kind"] == "order" and same(one["close"]["moves"][0]["reactions"], D.CANDIDATES["truth"])
      and one["close"]["moves"][0]["gain"] > 1000)
check("one step from the true network: no single change explains the data better (and no reaction is offered for adding, "
      "because every species is already explained)",
      one["truth"]["moves"][0]["gain"] < 0 and not any(m_["kind"] == "add" for m_ in one["truth"]["moves"]))
t = time.time()
one_sb = D.one_step_revision(no_E, d, inhib=np.full(len(d["gc"]), 0.03))
n_sb = sum(m_["site_blocking"] for m_ in one_sb["moves"])
print(f"     one step from 'no E' with site blocking (K_C = 0.03 /bar) as a second footing: best is to {one_sb['moves'][0]['what']}; "
      f"{n_sb} of {one_sb['n_tried']} changes fitted better with it ({time.time() - t:.0f} s)")
check("one step with site blocking at a given strength as a second footing (what the advisor passes when its own fit found "
      "site blocking): the same changes are tried, each says which footing its fit used, and from the network without "
      "C + B -> E the best change still adds it",
      one_sb["n_tried"] == one["no E"]["n_tried"] and all(isinstance(m_["site_blocking"], bool) for m_ in one_sb["moves"])
      and one_sb["moves"][0]["kind"] == "add" and same(one_sb["moves"][0]["reactions"], D.CANDIDATES["truth"])
      and isinstance(one_sb["current"]["site_blocking"], bool))
all_moves = [m_ for r_ in one.values() for m_ in r_["moves"]]
check("every change offered is a single one of the four kinds, a valid network of at most 4 reactions, listed best first; "
      "the best three are fitted to convergence; JSON-friendly",
      all(m_["kind"] in ("add", "remove", "order", "replace") and 1 <= len(D.parse_network(m_["reactions"])) <= 4 for m_ in all_moves)
      and all([m_["fit"]["bic"] for m_ in r_["moves"]] == sorted(m_["fit"]["bic"] for m_ in r_["moves"]) for r_ in one.values())
      and all(sum(m_["converged"] for m_ in r_["moves"]) == 3 and all(m_["converged"] for m_ in r_["moves"][:3]) for r_ in one.values())
      and all(len({D.network_key(D.parse_network(m_["reactions"])) for m_ in r_["moves"]}) == r_["n_tried"] for r_ in one.values())
      and len(json.dumps(one["close"])) > 100)

# ----------------------------------------------------------------------------------- 7. command line
out = subprocess.run([sys.executable, os.path.join(HERE, "discover_reactions.py"), os.path.join(EX, "complex_noisy.csv"),
                      "--candidates", os.path.join(EX, "candidates_truth.txt"), os.path.join(EX, "candidates_far.txt"),
                      "--no-site-blocking", "--quiet", "--next-run"], capture_output=True, text=True, cwd=HERE)
check("discover_reactions.py --candidates ... --next-run prints the ranking and the run",
      out.returncode == 0 and "The data support 'truth'." in out.stdout and "cannot explain E" in out.stdout and "Run at" in out.stdout)
bad = os.path.join(os.environ.get("TEMP", HERE), "bad_candidate_test_discovery.txt")
with open(bad, "w") as fh:
    fh.write("A + B -> E\n")
out = subprocess.run([sys.executable, os.path.join(HERE, "discover_reactions.py"), os.path.join(EX, "complex_noisy.csv"),
                      "--candidates", bad], capture_output=True, text=True, cwd=HERE)
os.remove(bad)
check("discover_reactions.py explains an unbalanced candidate in plain words, naming the file, instead of a traceback",
      out.returncode != 0 and "not balanced" in out.stderr and "bad_candidate_test_discovery.txt" in out.stderr
      and "Traceback" not in out.stderr)
t = time.time()
out = subprocess.run([sys.executable, os.path.join(HERE, "discover_reactions.py"), os.path.join(EX, "complex_noisy.csv"),
                      "--candidates", os.path.join(EX, "candidates_truth.txt"), "--quiet",
                      "--json", os.path.join(HERE, "no_such_folder_test_discovery", "out.json")], capture_output=True, text=True, cwd=HERE)
check("discover_reactions.py --json into a folder that does not exist says so in plain words before any computing, "
      "instead of a traceback after it",
      out.returncode != 0 and "does not exist" in out.stderr and "Traceback" not in out.stderr and time.time() - t < 30
      and not os.path.exists(os.path.join(HERE, "no_such_folder_test_discovery")))

if QUICK:
    print(f"\nAll quick reaction-discovery checks passed ({time.time() - T0:.0f} s; the searches were skipped).")
    sys.exit(0)

# ----------------------------------------------------------------------------------- 8. the automatic search
def ends_at(res, truth):
    """'exact' if the search returns the true network, 'tied' if the true network is among the
    runners-up the search itself reports as indistinguishable from its best, else 'no'."""
    if same(res["best"]["reactions"], truth):
        return "exact"
    return "tied" if any(same(f_["reactions"], truth) and f_["tied_with_best"] for f_ in res["runners_up"]) else "no"


seen = []
res = D.discover(data["mystery_noisy.csv"], progress=lambda info: seen.append(info["stage"]))
how = ends_at(res, M.MYSTERY_TEXT)
print(f"     mystery_noisy.csv: {'; '.join(res['best']['reactions'])}   [{how}]  {res['n_fits']} fits, {res['seconds']:.0f} s")
print("     left open: " + (" ".join(res["undecided"]) or "nothing"))
check("automatic search on mystery_noisy.csv recovers the mystery network (exactly, or tied with it and said so)",
      how in ("exact", "tied") and res["best"]["fits_noise"] and res["complete"])
check("the search reports its path, calls the progress callback and returns JSON-friendly results",
      len(res["path"]) >= 4 and "search" in seen and "final" in seen and len(json.dumps(res)) > 100)
kf = {D.parse_reaction(r_["reaction"]).direction_key(): r_ for r_ in res["best"]["rate_constants"]}
ok = all(abs(kf[r_.direction_key()]["k_ref"] / k_ - 1) < 0.10 and abs(kf[r_.direction_key()]["Ea_kJ"] - e_ / 1000) < 5
         for r_, k_, e_ in zip(D.parse_network(M.MYSTERY_TEXT), M.MYSTERY_K, M.MYSTERY_EA) if r_.direction_key() in kf)
check("mystery network: fitted rate constants within 10 % and activation energies within 5 kJ/mol of the truth", ok and len(kf) == 3)

for start in ("far", "similar", "close"):
    res = D.discover(data["complex_noisy.csv"], start=D.CANDIDATES[start], models=("plain",), n_final=2)
    how = ends_at(res, D.CANDIDATES["truth"])
    print(f"     from '{start}': {'; '.join(res['best']['reactions'])}   [{how}]  {res['n_fits']} fits, {res['seconds']:.0f} s")
    check(f"starting from '{start}', the search ends at the true complex network (exactly, or tied with it and said so)",
          how in ("exact", "tied") and res["best"]["fits_noise"])

# A short optimisation campaign instead of a kinetic study: the 5 shared screening runs and 3 more
# allowed steps on the virtual 'real' reactor (site blocking + GC noise).  Here the first pass of the
# search takes a wrong first reaction and ends at a network that does not fit; it must notice
# and start again.  Run with the default settings, as a user would (site blocking allowed).
short = D.load_data(os.path.join(EX, M.SHORT))
Xc = short["phys"][:, :5]
check("complex_short_campaign.csv reproduces from its seed; its 8 runs start with the 5 shared screening runs, respect "
      "the step limits and the box, and keep the catalyst mass",
      np.max(np.abs(fresh[M.SHORT]["gc"] - short["gc"])) < 2e-6 and np.allclose(fresh[M.SHORT]["phys"], short["phys"])
      and len(Xc) == 8 and np.allclose(Xc[:5], screening_runs(base_case(0)))
      and np.all(np.abs(np.diff(Xc, axis=0)) <= MAX_STEP + 1e-9) and np.all(Xc >= LOW) and np.all(Xc <= HIGH)
      and np.all(short["phys"][:, 5] == 1.0))
res = D.discover(short)
how = ends_at(res, D.CANDIDATES["truth"])
print(f"     8 step-limited runs: {'; '.join(res['best']['reactions'])}   [{how}]  {res['restarts']} restart(s), {res['n_fits']} fits, {res['seconds']:.0f} s")
check("short step-limited campaign: the first pass ends in a dead end, the search says so, starts again and ends at the true network",
      res["restarts"] >= 1 and any("search again" in p_ for p_ in res["path"]) and how in ("exact", "tied") and res["best"]["fits_noise"])
# Like for like.  On these runs the best network keeps site blocking (one more fitted number).  The
# runners-up must have been given the same chance, otherwise the BIC differences reported would make
# the best network look more clearly ahead than it is (it once did: 10.9 reported against 9.2).
near = [f_ for f_ in res["runners_up"] if f_["delta_bic"] < 30]
direct = [D.fit_network(f_["reactions"], short, n_starts=1)["bic"] for f_ in near]
print("     runners-up, BIC reported by the search / fitted directly with site blocking allowed: "
      + ", ".join(f"{f_['bic']:.1f} / {b_:.1f}" for f_, b_ in zip(near, direct)))
check("the BIC differences the search reports compare like with like: the best network keeps site blocking here, and no "
      "runner-up close to it is listed with a higher BIC than a direct fit that allows site blocking gives",
      res["best"]["model"] == "inhib" and len(near) >= 1 and all(f_["bic"] <= b_ + 1.0 for f_, b_ in zip(near, direct)))

# A true rate law that does NOT follow its equation: the network of candidates_close.txt taken as the
# truth (C + B -> E at r = k P_C, zero order in B).  The search is not told; it must find the order,
# and must not keep a reaction it added on the way to make up for the order it had not found yet
# (it once returned such a network, 15 BIC units worse than the true one, as "fits").
zero = D.load_data(os.path.join(EX, M.ZERO))
check(f"{M.ZERO} reproduces from the network of candidates_close.txt and its seed",
      np.max(np.abs(fresh[M.ZERO]["gc"] - zero["gc"])) < 2e-6 and np.allclose(fresh[M.ZERO]["phys"], zero["phys"])
      and same(M.ZERO_TEXT, D.CANDIDATES["close"]))
res = D.discover(zero, models=("plain",), n_final=2)
how = ends_at(res, M.ZERO_TEXT)
true_fit = D.fit_network(M.ZERO_TEXT, zero, models=("plain",), n_starts=2)
print(f"     zero-order truth: {'; '.join(res['best']['reactions'])}   [{how}]  {res['restarts']} restart(s), {res['n_fits']} fits, {res['seconds']:.0f} s;"
      f"  BIC {res['best']['bic']:.1f} (true network fitted directly: {true_fit['bic']:.1f})")
check("a true rate that is zero order in a reactant: the search finds the order and ends at the true network "
      "(exactly, or tied with it and said so), with no reaction left over",
      how in ("exact", "tied") and res["best"]["fits_noise"] and res["best"]["n_reactions"] == 3
      and res["best"]["bic"] < true_fit["bic"] + D.INDISTINGUISHABLE)

# a time limit is a limit on the whole search, the final check included (a web page can rely on it)
noE30 = D.make_data(data["complex_noisy.csv"]["phys"],
                    np.column_stack([data["complex_noisy.csv"]["gc"][:, :4], np.zeros(30), data["complex_noisy.csv"]["gc"][:, 5]]),
                    name="E not logged")
t = time.time()
res = D.discover(noE30, max_seconds=10)
took = time.time() - t
print(f"     discover(max_seconds=10) on a log nothing fits: {took:.1f} s, stopped early: {res['stopped_early']}")
check("discover(max_seconds=...) bounds the whole search: it stops close to the limit (one batch of fits may run past it), "
      "says so, and still returns its best network",
      res["stopped_early"] and took < 25 and any("time limit" in s_ for s_ in res["summary"]) and res["best"]["reactions"])

# far too few runs: the first two rows of complex_noisy.csv.  Many networks fit two readings; the search must
# still return one, say that the data cannot decide, and warn that this is a first guess (no crash, no claim)
dn2 = data["complex_noisy.csv"]
d2 = D.make_data(dn2["phys"][:2], dn2["gc"][:2], dn2["hours"][:2], name="two runs")
res = D.discover(d2)
print(f"     two runs only: {'; '.join(res['best']['reactions'])}  ({len(res['undecided'])} undecided, {res['seconds']:.0f} s)")
check("two runs only: the search returns a network, says what the data cannot decide, and warns that so few runs give a "
      "first guess only",
      res["best"]["n_reactions"] >= 1 and not res["decided"] and len(res["undecided"]) >= 1
      and any("only 2 runs" in s for s in res["summary"]) and len(json.dumps(CLI.strip(res))) > 50)

if "--thorough" in sys.argv:
    # The same zero-order network on ANOTHER run design (design seed 1, noise seed 1; not shipped as a
    # file).  Here the first pass and the first new start both end in dead ends, because they take
    # every rate to follow its equation until the reactions are settled.  The second new start, which
    # also tries every reaction it may add with one order changed, must find the true network.
    phys1 = M.study_design(seed=1)
    zero1 = D.make_data(phys1, M.noisy_gc(M.exact_gc(M.zero_order_system(), phys1), 1), name="zero order, run design 1")
    res = D.discover(zero1)
    how = ends_at(res, M.ZERO_TEXT)
    print(f"     zero-order truth, another run design: {'; '.join(res['best']['reactions'])}   [{how}]  {res['restarts']} restart(s), "
          f"{res['n_fits']} fits, {res['seconds']:.0f} s")
    check("thorough: a zero-order rate on another run design is missed by the first pass and the first new start, and found "
          "by the second new start, which also varies the orders of the reactions it adds",
          how in ("exact", "tied") and res["best"]["fits_noise"] and res["restarts"] == 2
          and any("one of its orders changed" in p_ for p_ in res["path"]))

print(f"\nAll reaction-discovery checks passed ({time.time() - T0:.0f} s).")
