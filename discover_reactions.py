"""
Which reactions are happening in the bed?  Command-line front end of smart/discovery.py.

    python discover_reactions.py examples/discovery/mystery_noisy.csv
        automatic search: starts from no reactions and builds the network that explains the GC
        readings best (1 to 2 minutes for 30 runs)

    python discover_reactions.py examples/discovery/complex_noisy.csv --candidates examples/discovery/candidates_truth.txt examples/discovery/candidates_close.txt examples/discovery/candidates_similar.txt examples/discovery/candidates_far.txt
        ranks candidate networks you typed (one text file per candidate)

    python discover_reactions.py examples/discovery/complex_noisy.csv --start examples/discovery/candidates_far.txt
        repairs a starting network: adds, removes and replaces reactions until nothing improves

    add  --next-run   to print the experiment that would best tell the top two networks apart
         (a free run anywhere in the operating box, as in a kinetic study; with --from-last-run
         it stays within one allowed step of the last run in the log, as the next run of a
         campaign must)

The run log is the CSV the website and reactor_advisor.py use:
    pressure_bar, flow_A_mL_min, flow_B_mL_min, flow_I_mL_min, temperature_C, catalyst_g,
    gc_A, gc_B, gc_C, gc_D, gc_E, gc_I [, hours_h]

A candidate file holds one reaction per line, for example
    A + B <=> C
    C + B -> E ; orders C=1, B=0
    A -> D
'->' runs one way, '<=>' both ways (to equilibrium).  The rate follows the equation unless
'orders' says otherwise.  Species: A and B are fed, C = AB, D = A rearranged, E = AB2, I = inert.

Words used in the output
    GC noise     how much a GC mole fraction scatters by itself: 1 % of the reading + 0.05 mol-%
    typical miss root-mean-square difference between the model and the GC over all runs
    BIC          misfit in units of the GC noise + a penalty for every fitted number; lower is
                 better; a difference below 6 means the data cannot tell two networks apart
ALL KINETIC NUMBERS IN THE EXAMPLES ARE PLACEHOLDERS.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from smart import discovery as D
from smart.systems import A_SPEC


# =================================================================================== plain-word reports
def lines_fit(f, indent="  "):
    """Equations, rate constants and misses of one fitted network."""
    out = [indent + e for e in f["equations"]]
    out.append(indent + f"fitted with {f['model_words']}; {f['n_params']} fitted numbers, {f['n_runs']} runs")
    for j, r in enumerate(f["rate_constants"], 1):
        out.append(indent + f"k{j} at 300 C = {r['k_ref']:.3g} ({r['k_ref_low']:.3g} to {r['k_ref_high']:.3g}) {r['k_units']}"
                   f",   Ea{j} = {r['Ea_kJ']:.0f} +- {r['Ea_sd_kJ']:.0f} kJ/mol")
    out.append(indent + "typical miss per species: " + ",  ".join(
        f"{s} {m['mol_pct']:.2f} mol-% ({m['x_noise']:.1f}x noise)" for s, m in f["miss"].items()))
    return out


def report_ranking(res, source=""):
    """Plain-text lines (a list of str) reporting a rank_candidates result: the table of candidates,
    the verdict, what the data cannot decide, notes and cautions about the log, and the fitted numbers
    of the best candidate.  Needs the keys ranking, best, decided, undecided and n_runs (notes and
    cautions if present)."""
    out = [f"Which candidate network do the data support?   ({res['n_runs']} runs{', ' + source if source else ''})", ""]
    w = max(len(f["name"]) for f in res["ranking"])
    w = max(w, 9) + 2
    out.append(f"  {'rank':<5}{'candidate':<{w}}{'reactions':>9}{'BIC':>12}{'difference':>12}{'evidence':>10}   verdict")
    for f in res["ranking"]:
        out.append(f"  {f['rank']:<5}{f['name']:<{w}}{f['n_reactions']:>9}{f['bic']:>12.1f}{f['delta_bic']:>12.1f}"
                   f"{100 * f['weight']:>9.1f}%   {f['verdict']}")
    out.append("")
    fits = res["ranking"][0]["fits_noise"]
    if not fits:
        # never "the data support X" for a candidate that misses the data: it is only the least bad
        out.append(f"None of the candidates fits the data within the GC noise. '{res['best']}' is the least bad, but "
                   "something is happening that none of the candidates describes.")
    elif res["decided"]:
        out.append(f"The data support '{res['best']}'.")
    if not res["decided"]:
        out += res["undecided"]
    if "cautions" in res:
        out += res.get("notes", []) + res["cautions"]
    elif res["n_runs"] < D.FEW_RUNS:                 # a result saved by an older version
        out.append(D._few_runs_note(res["n_runs"]))
    out += ["", f"The {'best' if fits else 'least bad'} candidate, '{res['best']}':"] + lines_fit(res["ranking"][0])
    return out


def report_discovery(res, source=""):
    """Plain-text lines (a list of str) reporting a discover result: how the search went (path), the
    summary (which already holds the notes and cautions about the log), the fitted numbers of the best
    network and the runners-up.  Needs the keys n_fits, seconds, path, summary, best and runners_up."""
    out = [f"Reaction search{' on ' + source if source else ''}: {res['n_fits']} network fits in {res['seconds']:.0f} s", ""]
    out += ["How the search went:"] + ["  " + p for p in res["path"]] + [""]
    out += res["summary"] + [""]
    out += ["Fitted numbers of the best network:"] + lines_fit(res["best"]) + [""]
    if res["runners_up"]:
        out.append("The next best networks:")
        for f in res["runners_up"]:
            tie = "cannot be told apart from the best" if f["tied_with_best"] else "ruled out against the best"
            out.append(f"  BIC difference {f['delta_bic']:.1f} ({tie}):  {';  '.join(f['reactions'])}")
    return out


def report_next_run(nr):
    """Plain-text lines (a list of str) reporting a discriminating_run result: the run, the GC readings
    each network predicts there and what one such run would add to the BIC difference.  Needs the keys
    message, predicted, expected_bic_gap (and step_limited)."""
    out = ["The run that would best tell the top two networks apart:", "  " + nr["message"], "",
           "  predicted GC reading (mol-%):   " + "".join(f"{s:>8}" for s in next(iter(nr["predicted"].values())))]
    for name, y in nr["predicted"].items():
        out.append(f"  {name[:30]:<32}" + "".join(f"{v:>8.2f}" for v in y.values()))
    out.append(f"  allowing for the uncertainty of both fits, one such run would add about {nr['expected_bic_gap']:.0f} to the "
               "BIC difference (6 is enough to decide).")
    if not nr.get("step_limited"):
        out.append("  This is a free run, as in a kinetic study, and may be far from the last run in the log. If the reactor must "
                   "move one allowed step at a time, ask for the run within one step of the last run (--from-last-run).")
    return out


def strip(obj):
    """Drop the bulky parts of a result before saving it as JSON."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in ("cov", "predicted_gc")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


# =================================================================================== command line
def main():
    """The command line: read the log, rank candidates or search, print the report, optionally the
    next run and a JSON file.  Every expected problem ends with a plain sentence, not a traceback."""
    ap = argparse.ArgumentParser(description="Find out which reactions happen in the reactor from the GC log.",
                                 epilog="See the top of this file and examples/discovery/WORKED_EXAMPLE.md.")
    ap.add_argument("log", help="run log (CSV)")
    ap.add_argument("--candidates", nargs="+", metavar="FILE", help="candidate networks to rank (one text file each)")
    ap.add_argument("--start", metavar="FILE", help="a starting network to repair")
    ap.add_argument("--next-run", action="store_true", help="also print the run that would best tell the top two networks apart")
    ap.add_argument("--from-last-run", action="store_true",
                    help="with --next-run: stay within one allowed step of the last run in the log")
    ap.add_argument("--max-reactions", type=int, default=4, help="largest network the search may build (default 4)")
    ap.add_argument("--surface-area", type=float, default=A_SPEC, help="BET surface area of the catalyst, m2/g (default 100)")
    ap.add_argument("--no-site-blocking", action="store_true",
                    help="plain rate laws only (do not let product C block catalyst sites)")
    ap.add_argument("--json", metavar="FILE", help="also save the full result as JSON")
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    args = ap.parse_args()
    if args.json and not os.path.isdir(os.path.dirname(os.path.abspath(args.json))):
        raise SystemExit(f"Cannot save the result to {args.json}: the folder {os.path.dirname(os.path.abspath(args.json))} "
                         "does not exist. Create it first or choose another place.")

    try:
        data = D.load_data(args.log, a_spec=args.surface_area)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not read {args.log}: {e}")
    models = ("plain",) if args.no_site_blocking else D.MODELS
    source = os.path.basename(args.log)

    def progress(info):
        if not args.quiet:
            print(f"   ... {info['elapsed_s']:4.0f} s  {info['message'][:100]}", file=sys.stderr, flush=True)

    def read_network(path):
        """A reaction file, with the file named in the message when a reaction in it cannot be used."""
        try:
            return D.read_network_file(path)
        except D.ReactionError as e:
            raise D.ReactionError(f"in {os.path.basename(path)}: {e}") from None

    try:
        if args.candidates:
            cands = {}
            for p in args.candidates:
                name = os.path.splitext(os.path.basename(p))[0].replace("candidates_", "")
                if name in cands:
                    raise ValueError(f"two candidate files have the same name '{name}'; give every candidate its own file name.")
                cands[name] = read_network(p)
                if not cands[name]:
                    raise D.ReactionError(f"{os.path.basename(p)} holds no reaction (one reaction per line, for example 'A + B <=> C').")
            res = D.rank_candidates(cands, data, models=models, progress=progress)
            print("\n".join(report_ranking(res, source)))
        else:
            start = read_network(args.start) if args.start else None
            res = D.discover(data, start=start, max_reactions=args.max_reactions, models=models, progress=progress)
            print("\n".join(report_discovery(res, source)))
    except D.ReactionError as e:
        raise SystemExit(f"Problem with a reaction: {e}")
    except OSError as e:                             # a candidate or starting-network file that cannot be opened
        raise SystemExit(f"Could not read a reaction file: {e}")
    except ValueError as e:
        raise SystemExit(f"Cannot do this: {e}")
    out = dict(result=res)
    if args.next_run:
        try:
            nr = D.discriminating_run(res, data, x_last=data["phys"][-1, :5] if args.from_last_run else None)
            print("\n" + "\n".join(report_next_run(nr)))
            out["next_run"] = nr
        except ValueError as e:
            print(f"\nNo run to suggest: {e}")
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(strip(out), fh, indent=1)
        except OSError as e:
            raise SystemExit(f"Could not save the result to {args.json}: {e.strerror or e}.")
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
