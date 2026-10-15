"""
SMART REACTOR ADVISOR - tells a reactor operator what to run next, in plain English.

Who it is for
    Someone who runs a small catalytic reactor (a lab, a pilot plant, a small chemicals
    business) and wants to make more money from it, but cannot afford dozens of trial runs
    and does not know kinetics or machine learning.  Every trial run costs a day of work,
    feed gas and catalyst.

What they give it
    The run log they already keep: what they set (pressure, the two gas flows, temperature,
    catalyst mass) and what the GC printed at the outlet.  Nothing else.  Profit is worked out
    from the GC reading, using the inert gas as internal standard.

What it gives back
    1. how each run went, in money per hour
    2. what it learned about their chemistry, in plain words ("every +10 C makes the
       side reaction 1.6x faster", "above 290 C the equilibrium caps your conversion")
    3. the next settings to try, as changes from their best run, with the expected profit
       and a likely range, and WHY that run
    4. whether another trial is worth its cost
    5. the best proven settings to use today
    It saves the suggestion to next_run.csv.

How it decides (see smart/advisor.py): it knows the physics of a packed-bed reactor and the
reaction network, fits the unknown rate constants to the GC readings, corrects the physics
with a Gaussian process where the data disagree with it, and picks the run with the highest
expected improvement.

Usage
    python reactor_advisor.py --demo                      # make a 6-run example log and advise on it
    python reactor_advisor.py my_reactor_log.csv          # advise on your own log
    python reactor_advisor.py my_reactor_log.csv --system simple
    python reactor_advisor.py my_reactor_log.csv --limit temperature=200:300 --fix pressure=30
    python reactor_advisor.py my_reactor_log.csv --trial-cost 50 --hours 6000
    python reactor_advisor.py --demo --simulate-days 8    # watch it run your (virtual) reactor for 8 days

Log format (CSV, one row per run; GC in mole fractions or %):
    pressure_bar, flow_A_mL_min, flow_B_mL_min, temperature_C, catalyst_g, gc_A, gc_B, gc_C, ..., gc_I
    The inert flow is 10 mL/min in every run (as in the rest of the project).

Safety: the advisor only SUGGESTS.  Check every suggestion against your plant's safety limits
(use --limit to keep it inside them) before running it.  Prices are placeholders: replace them.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import argparse
import csv
import warnings
import numpy as np

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from smart.systems import SYSTEMS, LOW, HIGH, V_I, R, unscale, scale
from smart.advisor import SmartAdvisor, VirtualReactor, profit_from_gc

# friendly knob names in the order the optimiser uses (P, V_A, V_B, T, Wcat)
KNOBS = ["pressure", "flow_A", "flow_B", "temperature", "catalyst"]
COLS = ["pressure_bar", "flow_A_mL_min", "flow_B_mL_min", "temperature_C", "catalyst_g"]
SHOW = ["Pressure", "Flow of A", "Flow of B", "Temperature", "Catalyst"]
SHOW_LC = ["pressure", "flow of A", "flow of B", "temperature", "catalyst"]
UNIT = ["bar", "mL/min", "mL/min", "°C", "g"]
FMT = ["{:.0f}", "{:.1f}", "{:.1f}", "{:.0f}", "{:.2f}"]
REACTION_NAMES = {
    "simple": ["main reaction  A + B -> C", "side reaction  A -> D"],
    "complex": ["main reaction  A + B <=> C", "over-reaction  C + B -> E (eats your product)",
                "cracking  A -> D"],
}


def to_internal(row_user):
    """User units (°C) -> optimiser units (K)."""
    p = np.array(row_user, float)
    p[..., 3] = p[..., 3] + 273.15
    return p


def to_user(phys):
    p = np.array(phys, float)
    p[..., 3] = p[..., 3] - 273.15
    return p


def fmt(i, v):
    return FMT[i].format(v) + " " + UNIT[i]


# =================================================================================== reading / writing logs
def read_log(path, S):
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if any(v.strip() for v in r.values())]
    need = COLS + [f"gc_{s}" for s in S.species]
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise SystemExit(f"{path} is missing columns {miss}.\nExpected: {', '.join(need)}")
    user = np.array([[float(r[c]) for c in COLS] for r in rows])
    gc = np.array([[float(r[f"gc_{s}"]) for s in S.species] for r in rows])
    if np.median(gc.sum(1)) > 1.5:          # given in % -> fractions
        gc = gc / 100.0
    gc = gc / gc.sum(1, keepdims=True)
    return user, gc


def write_log(path, S, user, gc):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run"] + COLS + [f"gc_{s}" for s in S.species])
        for i, (u, g) in enumerate(zip(user, gc), 1):
            w.writerow([i] + [round(v, 3) for v in u] + [round(v, 5) for v in g])


def make_demo_log(path, system, seed=0):
    """Six runs a careful operator might have done by hand (around the project's base case:
    10 bar, 20/20 mL/min, 600 K, 1 g), measured on the virtual reactor with GC noise."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed=seed + 7)
    hand = [[10, 20, 20, 327, 1.0], [10, 20, 20, 277, 1.0], [10, 20, 20, 377, 1.0],
            [10, 20, 20, 327, 2.0], [30, 20, 20, 327, 1.0], [10, 30, 20, 327, 1.0]]
    gc = np.array([vr.run(scale(to_internal(u))[0])["gc"] for u in hand])
    write_log(path, S, np.array(hand, float), gc)
    return path


# =================================================================================== explanations
def explain_chemistry(adv, S, x_next):
    """Plain-language summary of the fitted kinetics."""
    out = []
    th = adv.kin.samples(400, np.random.default_rng(1))
    Ea = th[:, 1::2] * R * 573.15 / 1000.0                       # kJ/mol for every draw
    T = unscale(x_next)[0][3]
    names = REACTION_NAMES.get(S.name, [f"reaction {j + 1}" for j in range(S.nr)])
    for j, name in enumerate(names[:S.nr]):
        e_med, e_lo, e_hi = np.median(Ea[:, j]), *np.percentile(Ea[:, j], [10, 90])
        f10 = np.exp(e_med * 1000 / R * (1 / T - 1 / (T + 10)))
        out.append(f"   - {name}: activation energy about {e_med:.0f} kJ/mol "
                   f"(likely {e_lo:.0f}-{e_hi:.0f}); near {T - 273.15:.0f} °C every +10 °C "
                   f"makes it {f10:.2f}x faster.")
    if S.nr >= 2:
        e = np.median(Ea, 0)
        main = e[0]
        worse = [names[j].split("  ")[0] for j in range(1, S.nr) if e[j] > main]
        if worse:
            out.append(f"   - So heat helps the unwanted {' and '.join(worse)} more than the main "
                       f"reaction: hotter is faster, but you lose product. There is a best temperature.")
    if S.thermo[0] is not None:
        dH, dS = S.thermo[0]
        P, VA, VB, Tn, W = unscale(x_next)[0]
        r = V_I / max(VA, 1e-9)
        nA, nB = 1.0, VB / VA
        x = 0.9                                      # conversion of A capped below 90 % above T90
        tot = nA + nB - x + r
        K_need = x * tot / (P * (nA - x) * max(nB - x, 1e-3))
        T90 = -dH / (R * (np.log(K_need) - dS / R))
        out.append(f"   - The main reaction is reversible and gives off heat: at {P:.0f} bar, above about "
                   f"{T90 - 273.15:.0f} °C the equilibrium itself stops more than 90 % of A turning into C, "
                   f"however much catalyst you use.")
    fit = adv.kin
    resid = (fit.predict_gc(unscale(adv.X)) - adv.gc)[:, :-1]
    out.append(f"   - How well this chemistry explains your GC readings: typical miss "
               f"{100 * np.median(np.abs(resid)):.2f} percentage points of composition.")
    return out


def explain_move(best_phys, next_phys):
    """'Raise temperature by 15 °C, cut catalyst from 3.0 to 2.1 g' - changes vs the best run."""
    moves = []
    for i in range(5):
        a, b = best_phys[i], next_phys[i]
        rng = HIGH[i] - LOW[i]
        if abs(b - a) < 0.03 * rng:
            continue
        verb = "raise" if b > a else "lower"
        if i == 3:
            moves.append(f"{verb} {SHOW_LC[i]} by {abs(b - a):.0f} °C")
        else:
            moves.append(f"{verb} {SHOW_LC[i]} from {fmt(i, a)} to {fmt(i, b)}")
    return moves or ["keep the settings of your best run (the model wants to confirm it)"]


# =================================================================================== main
def advise(args, user, gc, S, quiet=False):
    phys = to_internal(user)
    X = scale(phys)
    profit = profit_from_gc(S, phys, gc)
    lo, hi, fixed = np.zeros(5), np.ones(5), {}
    for lim in args.limit:
        k, rng_ = lim.split("=")
        i = KNOBS.index(k)
        a, b = (float(v) for v in rng_.split(":"))
        if i == 3:
            a, b = a + 273.15, b + 273.15
        lo[i], hi[i] = (np.clip([a, b], LOW[i], HIGH[i]) - LOW[i]) / (HIGH[i] - LOW[i])
    for fx in args.fix:
        k, v = fx.split("=")
        i = KNOBS.index(k)
        v = float(v) + (273.15 if i == 3 else 0)
        fixed[i] = (v - LOW[i]) / (HIGH[i] - LOW[i])

    adv = SmartAdvisor(S, seed=args.seed, lo=lo, hi=hi, fixed=fixed)
    adv.update(X, gc, profit)
    x_next, info = adv.suggest(np.random.default_rng(args.seed))
    ib = adv.recommend()
    mu_data, sd_data = adv.predict(X)
    return adv, X, profit, x_next, info, ib, mu_data, sd_data


def report(args, user, gc, S, adv, X, profit, x_next, info, ib, mu_data):
    line = "=" * 78
    print(line)
    print("  SMART REACTOR ADVISOR")
    print(line)
    print(f"  Chemistry: {S.description}")
    print(f"  Your log : {len(profit)} runs from {args.log}.  Money = placeholder prices, per hour.\n")

    print("1. HOW YOUR RUNS WENT  (profit worked out from your GC readings)")
    print("   run " + "".join(f"{s[:11]:>12s}" for s in SHOW) + f"{'profit/h':>11s}")
    for i, (u, p) in enumerate(zip(user, profit), 1):
        tag = "   <- best" if i - 1 == ib else ""
        print(f"   {i:3d} " + "".join(f"{FMT[j].format(u[j]):>12s}" for j in range(5)) + f"{p:11.2f}{tag}")
    print(f"   Best proven run: run {ib + 1}, making about {mu_data[ib]:.1f} per hour "
          f"(the model's estimate, which smooths out GC noise).\n")

    print("2. WHAT I LEARNED ABOUT YOUR CHEMISTRY  (from the GC readings, not just profit)")
    for s in explain_chemistry(adv, S, x_next):
        print(s)
    print()

    nxt = unscale(x_next)[0]
    best = unscale(X[ib])[0]
    gain = info["pred"] - mu_data[ib]
    lo95, hi95 = info["pred"] - 1.64 * info["sd"], info["pred"] + 1.64 * info["sd"]
    print("3. WHAT TO RUN NEXT")
    print("   " + ",  ".join(f"{SHOW[i]} {fmt(i, to_user(nxt)[i])}" for i in range(5)))
    print("   Compared with your best run: " + "; ".join(explain_move(to_user(best), to_user(nxt))) + ".")
    print(f"   Expected profit: {info['pred']:.1f} per hour (likely between {lo95:.1f} and {hi95:.1f}).")
    _, _, parts = adv.predict(x_next[None], parts=True)
    if gain > 1.64 * info["sd"]:
        why = (f"the chemistry model is fairly sure this beats your best run by about {gain:.1f} per hour.")
    elif parts["sd_phys"][0] > parts["sd_corr"][0]:
        why = ("the versions of your chemistry that fit your data so far disagree most about this "
               "region, and it could be better than your best run; this run settles it.")
    else:
        why = ("your data and the textbook physics disagree a little around here; this run checks "
               "whether the extra profit the model sees is real.")
    print(f"   Why this run: {why}\n")

    print("4. IS ANOTHER TRIAL WORTH IT?")
    ei = info["ei"]
    print(f"   Expected gain from this one trial: +{ei:.2f} per hour over your best run "
          f"(averaged over what could happen).")
    if args.trial_cost is not None:
        yearly = ei * args.hours
        verdict = "YES - run it." if yearly > args.trial_cost else "PROBABLY NOT - you are close to the best."
        print(f"   Over {args.hours:,.0f} operating hours a year that is about {yearly:,.0f}, against a trial cost of "
              f"{args.trial_cost:,.0f}: {verdict}")
    elif ei < 0.02 * max(abs(mu_data[ib]), 1):
        print("   That is under 2 % of your current profit: you are close to the best these knobs allow.")
    else:
        print("   Worth running. (Use --trial-cost and --hours to weigh it in money.)")
    print()
    print("5. IF YOU HAVE TO RUN THE PLANT TODAY")
    print("   Use your best proven settings (run {}): ".format(ib + 1) +
          ",  ".join(f"{fmt(i, to_user(best)[i])}" for i in range(5)) + ".")
    print(line)
    print("  Suggestions only: check them against your safety limits before running.")
    with open("next_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS + ["expected_profit_per_h", "low", "high"])
        w.writerow([round(v, 3) for v in to_user(nxt)] + [round(info["pred"], 2), round(lo95, 2), round(hi95, 2)])
    print("  Saved next_run.csv")


def simulate_days(args, S, user, gc):
    """Let the advisor run the (virtual) reactor for N days: each day it suggests a run, the
    virtual reactor + GC produce a reading, the log grows.  Shows how it would be used."""
    vr = VirtualReactor(S, seed=args.seed + 99)
    print(f"Running the advisor on the virtual {S.name} reactor for {args.simulate_days} days ...\n")
    print("  day  " + "".join(f"{s[:11]:>12s}" for s in SHOW) + f"{'expected':>10s}{'measured':>10s}{'best so far':>13s}")
    for d in range(1, args.simulate_days + 1):
        adv, X, profit, x_next, info, ib, mu_data, _ = advise(args, user, gc, S)
        r = vr.run(x_next)
        user = np.vstack([user, to_user(unscale(x_next))])
        gc = np.vstack([gc, r["gc"]])
        prof = profit_from_gc(S, to_internal(user), gc)
        print(f"  {d:3d}  " + "".join(f"{FMT[j].format(to_user(unscale(x_next))[0][j]):>12s}" for j in range(5))
              + f"{info['pred']:10.1f}{r['profit']:10.1f}{prof.max():13.1f}", flush=True)
    write_log(args.log, S, user, gc)
    print(f"\nThe log {args.log} now has {len(user)} runs. Final advice:\n")
    return user, gc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="my_reactor_log.csv", help="your run log (CSV)")
    ap.add_argument("--system", default="complex", choices=list(SYSTEMS),
                    help="which reaction network your reactor runs (default: complex)")
    ap.add_argument("--demo", action="store_true", help="create a 6-run example log first")
    ap.add_argument("--simulate-days", type=int, default=0,
                    help="let the advisor run the virtual reactor for this many days first")
    ap.add_argument("--limit", action="append", default=[],
                    help="safety range, e.g. --limit temperature=200:300 (°C) or --limit pressure=1:40")
    ap.add_argument("--fix", action="append", default=[], help="knob you cannot change, e.g. --fix catalyst=2")
    ap.add_argument("--trial-cost", type=float, default=None, help="cost of one trial run (money)")
    ap.add_argument("--hours", type=float, default=8000, help="operating hours per year (default 8000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    S = SYSTEMS[args.system]
    if args.demo:
        make_demo_log(args.log, args.system, args.seed)
        print(f"Wrote an example log {args.log}: 6 runs an operator might do by hand around 10 bar, "
              f"20 mL/min each, 327 °C, 1 g.\n")
    if not os.path.exists(args.log):
        raise SystemExit(f"{args.log} not found. Try:  python reactor_advisor.py --demo")
    user, gc = read_log(args.log, S)
    if args.simulate_days:
        user, gc = simulate_days(args, S, user, gc)
    adv, X, profit, x_next, info, ib, mu_data, _ = advise(args, user, gc, S)
    report(args, user, gc, S, adv, X, profit, x_next, info, ib, mu_data)


if __name__ == "__main__":
    main()
