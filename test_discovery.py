"""Checks for reaction discovery (smart/discovery.py, discover_reactions.py, examples/discovery/).
Run:  python test_discovery.py           about 6 min on a laptop (it runs five full reaction searches,
                                       35 to 90 s each); longer when the laptop is busy
      python test_discovery.py --quick   skips the searches: about 1.5 min"""
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
      and D.parse_reaction("A + B <-> C") == D.parse_reaction("A + B <=> C"))
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
noise_z = (data["complex_noisy.csv"]["gc"] - data["complex_exact.csv"]["gc"]) / D.gc_noise(data["complex_exact.csv"]["gc"])
check("the noise in the noisy file is of the size of the GC noise model (typical 0.5 to 1.1 of it)",
      0.5 < np.sqrt(np.mean(noise_z[:, :-1] ** 2)) < 1.1)
buf = io.StringIO(D.data_to_csv(data["complex_noisy.csv"]))
check("data_to_csv / load_data round trip (from text and from an open file)",
      np.allclose(D.load_data(buf.getvalue())["gc"], data["complex_noisy.csv"]["gc"], atol=1e-6)
      and np.allclose(D.load_data(buf)["phys"], data["complex_noisy.csv"]["phys"]))
check("detected products: C, D, E in the complex data", list(D.detected_species(data["complex_noisy.csv"])) == ["C", "D", "E"])

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

# ----------------------------------------------------------------------------------- 6. the run that tells two networks apart
d = data["complex_noisy.csv"]
nr = D.discriminating_run(ranks["complex_noisy.csv"], d)
x = np.array(nr["x"])
print("     " + nr["message"])
check("discriminating run (free): inside the operating box, enough flow for the GC, catalyst mass unchanged",
      np.all(x >= LOW - 1e-9) and np.all(x <= HIGH + 1e-9) and x[1:4].sum() >= MIN_TOTAL_FLOW and nr["run"]["catalyst_g"] == d["phys"][-1, 5])
check("discriminating run: truth and close differ there by far more than the GC noise, and more than in any run done",
      nr["networks"] == ["truth", "close"] and nr["separation"] > 10 and nr["separation"] > nr["best_so_far"])
x_last = d["phys"][-1, :5]
nr2 = D.discriminating_run(ranks["complex_noisy.csv"], d, x_last=x_last)
x2 = np.array(nr2["x"])
check("discriminating run (step-limited): within one allowed step of the last run and inside the box",
      np.all(np.abs(x2 - x_last) <= MAX_STEP + 1e-9) and np.all(x2 >= LOW - 1e-9) and np.all(x2 <= HIGH + 1e-9)
      and x2[1:4].sum() >= MIN_TOTAL_FLOW and nr2["step_limited"] and nr2["separation"] <= nr["separation"] + 1e-9)
yp = D.predict_gc(ranks["complex_noisy.csv"]["ranking"][0], np.r_[x, 1.0][None, :])[0]
check("predict_gc of a fitted network agrees with the prediction reported for the run; JSON-friendly",
      np.allclose(100 * yp, list(nr["predicted"]["truth"].values()), atol=1e-6) and len(json.dumps(nr)) > 50)

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
check("discover_reactions.py explains an unbalanced candidate in plain words instead of a traceback",
      out.returncode != 0 and "not balanced" in out.stderr and "Traceback" not in out.stderr)

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
# and start again.
short = D.load_data(os.path.join(EX, M.SHORT))
Xc = short["phys"][:, :5]
check("complex_short_campaign.csv reproduces from its seed; its 8 runs start with the 5 shared screening runs, respect "
      "the step limits and the box, and keep the catalyst mass",
      np.max(np.abs(fresh[M.SHORT]["gc"] - short["gc"])) < 2e-6 and np.allclose(fresh[M.SHORT]["phys"], short["phys"])
      and len(Xc) == 8 and np.allclose(Xc[:5], screening_runs(base_case(0)))
      and np.all(np.abs(np.diff(Xc, axis=0)) <= MAX_STEP + 1e-9) and np.all(Xc >= LOW) and np.all(Xc <= HIGH)
      and np.all(short["phys"][:, 5] == 1.0))
res = D.discover(short, models=("plain",), n_final=2)
how = ends_at(res, D.CANDIDATES["truth"])
print(f"     8 step-limited runs: {'; '.join(res['best']['reactions'])}   [{how}]  {res['restarts']} restart(s), {res['n_fits']} fits, {res['seconds']:.0f} s")
check("short step-limited campaign: the first pass ends in a dead end, the search says so, starts again and ends at the true network",
      res["restarts"] >= 1 and any("search again" in p_ for p_ in res["path"]) and how in ("exact", "tied") and res["best"]["fits_noise"])

print(f"\nAll reaction-discovery checks passed ({time.time() - T0:.0f} s).")
