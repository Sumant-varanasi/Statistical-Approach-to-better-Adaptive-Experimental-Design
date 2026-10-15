"""
SMART REACTOR ADVISOR - tells a reactor operator what to run next, in plain English.

Who it is for
    Someone who runs a small catalytic fixed-bed reactor (a lab, a pilot plant, a small
    chemicals business) and wants to make more money from it, but cannot afford dozens of
    trial runs and does not know kinetics or machine learning.  Every run costs a day of work,
    feed gas and catalyst time.

How it fits the way a reactor is actually run
    * The catalyst is loaded once.  Its mass and BET surface area are campaign settings; the
      advisor never asks you to change the catalyst.
    * You can change five set points between runs: pressure, the flows of A, B and the inert,
      and temperature.  The inert is a real knob: it dilutes the feed and lowers every partial
      pressure.
    * Set points cannot jump between consecutive runs: by default at most 10 bar, 10 mL/min per
      flow and 30 C per run (--max-step to change).  Every suggestion is within one step of
      your LAST run, so it is always viable, and the advisor tells you how many runs away the
      predicted best settings are.
    * The GC needs at least 10 mL/min of total feed and 2 mL/min of inert (internal standard).

What they give it
    The run log they already keep: what they set and what the GC printed.  Profit is worked
    out from the GC reading using the inert as the internal standard.

What it gives back
    1. how each run went, in money per hour
    2. what it learned about the chemistry, in plain words ("every +10 C makes the
       side reaction 1.6x faster", "above 290 C the equilibrium caps your conversion")
    3. the next run (one allowed step from the last run), with expected profit, and WHY
    4. where the model expects the best profit and how many runs it takes to get there
    5. whether another run is worth its cost
    6. the best proven settings to use today
    It saves the suggestion to next_run.csv.

Usage
    python reactor_advisor.py --demo                       # 5 example screening runs, then advice
    python reactor_advisor.py my_reactor_log.csv           # advise on your own log
    python reactor_advisor.py my_reactor_log.csv --system simple --surface-area 120
    python reactor_advisor.py my_reactor_log.csv --limit temperature=200:300 --fix pressure=30
    python reactor_advisor.py my_reactor_log.csv --max-step temperature=20 --max-step pressure=5
    python reactor_advisor.py my_reactor_log.csv --trial-cost 500 --hours 6000
    python reactor_advisor.py --demo --simulate-days 10    # watch it run your (virtual) reactor for 10 days

Log format (CSV, one row per run, in the order they were run; GC in mole fractions or %):
    pressure_bar, flow_A_mL_min, flow_B_mL_min, flow_I_mL_min, temperature_C, catalyst_g,
    gc_A, gc_B, gc_C, gc_D, [gc_E,] gc_I

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

from smart.systems import (SYSTEMS, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, WCAT, A_SPEC, R, unscale, scale,
                           physical, base_case, screening_runs)
from smart.advisor import SmartAdvisor, VirtualReactor, profit_from_gc, kpis_from_gc

# knobs in the optimiser's order: P, V_A, V_B, V_I, T
KNOBS = ["pressure", "flow_A", "flow_B", "flow_I", "temperature"]
COLS = ["pressure_bar", "flow_A_mL_min", "flow_B_mL_min", "flow_I_mL_min", "temperature_C", "catalyst_g"]
SHOW = ["Pressure", "Flow of A", "Flow of B", "Inert flow", "Temperature"]
SHOW_LC = ["pressure", "flow of A", "flow of B", "inert flow", "temperature"]
UNIT = ["bar", "mL/min", "mL/min", "mL/min", "°C"]
FMT = ["{:.0f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.0f}"]
REACTION_NAMES = {
    "simple": ["main reaction  A + B -> C", "side reaction  A -> D"],
    "complex": ["main reaction  A + B <=> C", "over-reaction  C + B -> E (eats your product)",
                "cracking  A -> D"],
}


def to_internal(rows):
    """User units (°C) -> model units (K); works on (n, 5) knob rows or (n, 6) rows with catalyst."""
    p = np.array(rows, float)
    p[..., 4] = p[..., 4] + 273.15
    return p


def to_user(rows):
    p = np.array(rows, float)
    p[..., 4] = p[..., 4] - 273.15
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


def make_demo_log(path, system, wcat, a_spec, seed=0):
    """The 5 screening runs an operator does first (a rising temperature ladder, then pressure
    up, then more B), measured on the virtual reactor with GC noise."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed=seed + 7, wcat=wcat, a_spec=a_spec)
    rows = screening_runs(base_case(seed))
    gc = np.array([vr.run(scale(r)[0])["gc"] for r in rows])
    user = np.column_stack([to_user(rows), np.full(len(rows), wcat)])
    write_log(path, S, user, gc)
    return rows


# =================================================================================== explanations
def explain_chemistry(adv, S, x_ref):
    """Plain-language summary of the fitted kinetics."""
    out = []
    th = adv.kin.samples(400, np.random.default_rng(1))
    Ea = th[:, 1::2] * R * 573.15 / 1000.0                       # kJ/mol for every draw
    P, VA, VB, VI, T = unscale(x_ref)[0]
    names = REACTION_NAMES.get(S.name, [f"reaction {j + 1}" for j in range(S.nr)])
    for j, name in enumerate(names[:S.nr]):
        e_med, e_lo, e_hi = np.median(Ea[:, j]), *np.percentile(Ea[:, j], [10, 90])
        f10 = np.exp(e_med * 1000 / R * (1 / T - 1 / (T + 10)))
        out.append(f"   - {name}: activation energy about {e_med:.0f} kJ/mol "
                   f"(likely {e_lo:.0f}-{e_hi:.0f}); near {T - 273.15:.0f} °C every +10 °C "
                   f"makes it {f10:.2f}x faster.")
    if S.nr >= 2:
        e = np.median(Ea, 0)
        worse = [names[j].split("  ")[0] for j in range(1, S.nr) if e[j] > e[0]]
        if worse:
            out.append(f"   - So heat helps the unwanted {' and '.join(worse)} more than the main "
                       f"reaction: hotter is faster, but you lose product. There is a best temperature.")
    if S.thermo[0] is not None:
        dH, dS = S.thermo[0]
        r = VI / max(VA, 1e-9)
        nA, nB = 1.0, VB / VA
        x = 0.9                                      # conversion of A capped below 90 % above T90
        tot = nA + nB - x + r
        K_need = x * tot / (P * (nA - x) * max(nB - x, 1e-3))
        T90 = -dH / (R * (np.log(K_need) - dS / R))
        out.append(f"   - The main reaction is reversible and gives off heat: at {P:.0f} bar, above about "
                   f"{T90 - 273.15:.0f} °C the equilibrium itself stops more than 90 % of A turning into C, "
                   f"however much catalyst you use.")
    out.append("   - Diluting with inert lowers every partial pressure, which slows the reactions; it only "
               "pays when it improves selectivity.")
    resid = (adv.kin.predict_gc(physical(adv.X, adv.W)) - adv.gc)[:, :-1]
    out.append(f"   - How well this chemistry explains your GC readings: typical miss "
               f"{100 * np.median(np.abs(resid)):.2f} percentage points of composition.")
    return out


def explain_move(from_phys, to_phys):
    """'raise pressure from 10 bar to 20 bar; lower temperature by 15 °C' - changes between two settings."""
    moves = []
    for i in range(5):
        a, b = from_phys[i], to_phys[i]
        if abs(b - a) < 0.02 * (HIGH[i] - LOW[i]):
            continue
        verb = "raise" if b > a else "lower"
        if i == 4:
            moves.append(f"{verb} {SHOW_LC[i]} by {abs(b - a):.0f} °C to {b:.0f} °C")
        else:
            moves.append(f"{verb} {SHOW_LC[i]} from {fmt(i, a)} to {fmt(i, b)}")
    return moves or ["keep the settings of your last run (the model wants to confirm it)"]


# =================================================================================== main
def parse_knob_args(args):
    lo, hi, fixed = np.zeros(5), np.ones(5), {}
    for lim in args.limit:
        k, rng_ = lim.split("=")
        i = KNOBS.index(k)
        a, b = (float(v) for v in rng_.split(":"))
        if i == 4:
            a, b = a + 273.15, b + 273.15
        lo[i], hi[i] = (np.clip([a, b], LOW[i], HIGH[i]) - LOW[i]) / (HIGH[i] - LOW[i])
    for fx in args.fix:
        k, v = fx.split("=")
        i = KNOBS.index(k)
        v = float(v) + (273.15 if i == 4 else 0)
        fixed[i] = (v - LOW[i]) / (HIGH[i] - LOW[i])
    step = MAX_STEP.copy()
    for ms in args.max_step:
        k, v = ms.split("=")
        if k == "flow":
            step[1:4] = float(v)
        else:
            step[KNOBS.index(k)] = float(v)
    return lo, hi, fixed, step


def advise(args, user, gc, S):
    phys6 = to_internal(user)
    X = scale(phys6[:, :5])
    wcat_rows = phys6[:, 5]
    profit = profit_from_gc(S, phys6, gc)
    lo, hi, fixed, step = parse_knob_args(args)
    adv = SmartAdvisor(S, wcat=wcat_rows[-1], a_spec=args.surface_area, seed=args.seed,
                       lo=lo, hi=hi, fixed=fixed, max_step=step)
    adv.update(X, gc, profit, wcat_rows=wcat_rows)
    x_next, info = adv.suggest(X[-1], np.random.default_rng(args.seed))
    ib = adv.recommend()
    mu_data, sd_data = adv.predict(X)
    return adv, X, profit, x_next, info, ib, mu_data, sd_data


def report(args, user, gc, S, adv, X, profit, x_next, info, ib, mu_data):
    line = "=" * 78
    phys6 = to_internal(user)
    XA, SC, _ = kpis_from_gc(S, phys6, gc)
    print(line)
    print("  SMART REACTOR ADVISOR")
    print(line)
    print(f"  Chemistry: {S.description}")
    print(f"  Catalyst : {user[-1, 5]:.3g} g loaded, {args.surface_area:.0f} m2/g (fixed for this campaign)")
    print(f"  Your log : {len(profit)} runs from {args.log}.  Money = placeholder prices, per hour.")
    print(f"  Step limits per run: {MAX_STEP_TEXT(adv)}\n")

    print("1. HOW YOUR RUNS WENT  (worked out from your GC readings)")
    print("   run " + "".join(f"{s[:11]:>12s}" for s in SHOW) + f"{'conv.':>7s}{'select.':>8s}{'profit/h':>10s}")
    for i, (u, p) in enumerate(zip(user, profit), 1):
        tag = "   <- best" if i - 1 == ib else ""
        print(f"   {i:3d} " + "".join(f"{FMT[j].format(u[j]):>12s}" for j in range(5))
              + f"{XA[i - 1]:7.2f}{SC[i - 1]:8.2f}{p:10.2f}{tag}")
    print(f"   Best proven run: run {ib + 1}, making about {mu_data[ib]:.1f} per hour "
          f"(the model's estimate, which smooths out GC noise).\n")

    print("2. WHAT I LEARNED ABOUT YOUR CHEMISTRY  (from the GC readings, not just profit)")
    for s in explain_chemistry(adv, S, X[-1]):
        print(s)
    print()

    nxt = unscale(x_next)[0]
    last = unscale(X[-1])[0]
    best = unscale(X[ib])[0]
    opt = unscale(info["model_opt"])[0]
    gain = info["pred"] - mu_data[ib]
    lo90, hi90 = info["pred"] - 1.64 * info["sd"], info["pred"] + 1.64 * info["sd"]
    print("3. WHAT TO RUN NEXT  (one allowed step from your last run)")
    print("   " + ",  ".join(f"{SHOW[i]} {fmt(i, to_user(nxt)[i])}" for i in range(5)))
    print("   Compared with your last run: " + "; ".join(explain_move(to_user(last), to_user(nxt))) + ".")
    print(f"   Expected profit: {info['pred']:.1f} per hour (likely between {lo90:.1f} and {hi90:.1f}).")
    _, _, parts = adv.predict(x_next[None], parts=True)
    if info.get("mode") == "walk":
        why = ("nothing within one step beats your best run yet, but the model is confident a much better "
               "region exists (see 4): this is the biggest allowed step toward it.")
    elif gain > 1.64 * info["sd"]:
        why = f"the chemistry model is fairly sure this beats your best run by about {gain:.1f} per hour."
    elif info["toward_opt"]:
        why = "it is the biggest allowed step toward where the model expects the best profit (see 4)."
    elif parts["sd_phys"][0] > parts["sd_corr"][0]:
        why = ("the versions of your chemistry that fit your data so far disagree most about this "
               "region, and it could be better than your best run; this run settles it.")
    else:
        why = ("your data and the textbook physics disagree a little around here; this run checks "
               "whether the extra profit the model sees is real.")
    print(f"   Why this run: {why}\n")

    print("4. WHERE THIS IS HEADING")
    print("   The model expects the best profit at: " +
          ",  ".join(f"{SHOW[i]} {fmt(i, to_user(opt)[i])}" for i in range(5)))
    print(f"   about {info['model_opt_pred']:.1f} per hour (likely between "
          f"{info['model_opt_pred'] - 1.64 * info['model_opt_sd']:.1f} and "
          f"{info['model_opt_pred'] + 1.64 * info['model_opt_sd']:.1f}), "
          f"{info['steps_away']} run(s) away under your step limits.")
    print("   This will firm up as runs come in: every GC reading sharpens the kinetics.\n")

    print("5. IS ANOTHER RUN WORTH IT?")
    ei = info["ei"]
    far = max(0.0, info["model_opt_pred"] - 1.64 * info["model_opt_sd"] - mu_data[ib])
    print(f"   Expected gain from this one run: +{ei:.2f} per hour over your best run "
          f"(averaged over what could happen).")
    print(f"   Expected gain from reaching the predicted best (conservative): +{far:.1f} per hour.")
    if args.trial_cost is not None:
        yearly = far * args.hours
        n = max(info["steps_away"], 1)
        cost = args.trial_cost * n
        verdict = ("YES - run it." if yearly > cost
                   else "PROBABLY NOT - the remaining gain does not pay for the runs needed.")
        print(f"   Over {args.hours:,.0f} operating hours a year that is about {yearly:,.0f}, against "
              f"{n} run(s) at {args.trial_cost:,.0f} each = {cost:,.0f}: {verdict}")
    elif far < 0.02 * max(abs(mu_data[ib]), 1):
        print("   That is under 2 % of your current profit: you are close to the best these knobs allow.")
    else:
        print("   Worth running. (Use --trial-cost and --hours to weigh it in money.)")
    print()
    print("6. IF YOU HAVE TO RUN THE PLANT TODAY")
    print(f"   Use your best proven settings (run {ib + 1}): " +
          ",  ".join(f"{fmt(i, to_user(best)[i])}" for i in range(5)) + ".")
    print(line)
    print("  Suggestions only: check them against your safety limits before running.")
    with open("next_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS + ["expected_profit_per_h", "low", "high"])
        w.writerow([round(v, 3) for v in to_user(nxt)] + [user[-1, 5], round(info["pred"], 2),
                                                           round(lo90, 2), round(hi90, 2)])
    print("  Saved next_run.csv")


def MAX_STEP_TEXT(adv):
    s = adv.step * (HIGH - LOW)
    return f"{s[0]:.0f} bar, {s[1]:.0f}/{s[2]:.0f}/{s[3]:.0f} mL/min (A/B/inert), {s[4]:.0f} °C"


def simulate_days(args, S, user, gc):
    """Let the advisor run the (virtual) reactor for N days: each day it suggests a run, the
    virtual reactor + GC produce a reading, the log grows.  Shows how it would be used."""
    vr = VirtualReactor(S, seed=args.seed + 99, wcat=user[-1, 5], a_spec=args.surface_area)
    print(f"Running the advisor on the virtual {S.name} reactor for {args.simulate_days} days "
          f"(one run per day, step limits respected) ...\n")
    print("  day  " + "".join(f"{s[:11]:>12s}" for s in SHOW) + f"{'expected':>10s}{'measured':>10s}{'best so far':>13s}{'best is':>10s}")
    for d in range(1, args.simulate_days + 1):
        adv, X, profit, x_next, info, ib, mu_data, _ = advise(args, user, gc, S)
        r = vr.run(x_next)
        user = np.vstack([user, np.r_[to_user(unscale(x_next))[0], user[-1, 5]]])
        gc = np.vstack([gc, r["gc"]])
        prof = profit_from_gc(S, to_internal(user), gc)
        print(f"  {d:3d}  " + "".join(f"{FMT[j].format(to_user(unscale(x_next))[0][j]):>12s}" for j in range(5))
              + f"{info['pred']:10.1f}{r['profit']:10.1f}{prof.max():13.1f}"
              + f"{info['steps_away']:5d} away", flush=True)
    write_log(args.log, S, user, gc)
    print(f"\nThe log {args.log} now has {len(user)} runs. Final advice:\n")
    return user, gc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="my_reactor_log.csv", help="your run log (CSV)")
    ap.add_argument("--system", default="complex", choices=list(SYSTEMS),
                    help="which reaction network your reactor runs (default: complex)")
    ap.add_argument("--surface-area", type=float, default=A_SPEC, help="BET area of the catalyst, m2/g")
    ap.add_argument("--catalyst", type=float, default=WCAT, help="catalyst mass loaded, g (used by --demo)")
    ap.add_argument("--demo", action="store_true", help="create an example log of 5 screening runs first")
    ap.add_argument("--simulate-days", type=int, default=0,
                    help="let the advisor run the virtual reactor for this many days first")
    ap.add_argument("--max-step", action="append", default=[],
                    help="largest change per run, e.g. --max-step temperature=20 --max-step flow=5 --max-step pressure=5")
    ap.add_argument("--limit", action="append", default=[],
                    help="safety range, e.g. --limit temperature=200:300 (°C) or --limit pressure=1:40")
    ap.add_argument("--fix", action="append", default=[], help="knob you cannot change, e.g. --fix flow_I=10")
    ap.add_argument("--trial-cost", type=float, default=None, help="cost of one run (money)")
    ap.add_argument("--hours", type=float, default=8000, help="operating hours per year (default 8000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    S = SYSTEMS[args.system]
    if args.demo:
        rows = make_demo_log(args.log, args.system, args.catalyst, args.surface_area, args.seed)
        b = to_user(rows[1])
        print(f"Wrote an example log {args.log}: the 5 screening runs an operator does first "
              f"(temperature ladder around {b[4]:.0f} °C at {b[0]:.0f} bar, then pressure up, then more B), "
              f"with {args.catalyst:g} g of catalyst loaded.\n")
    if not os.path.exists(args.log):
        raise SystemExit(f"{args.log} not found. Try:  python reactor_advisor.py --demo")
    user, gc = read_log(args.log, S)
    if args.simulate_days:
        user, gc = simulate_days(args, S, user, gc)
    adv, X, profit, x_next, info, ib, mu_data, _ = advise(args, user, gc, S)
    report(args, user, gc, S, adv, X, profit, x_next, info, ib, mu_data)


if __name__ == "__main__":
    main()
