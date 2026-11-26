"""
Builds examples/discovery/showcase/finder_showcase.xlsx from showcase_results.json (written by
make_finder_showcase.py).  Run `python make_finder_showcase.py --sheet` to rebuild it.

Sheets
    Read me            what the data sheet is, how the data were made, how to read it, the GC noise model (inputs)
    Advisor versions   v5, v6, v7 in short, and the terms used
    Cases              one row per case: true reactions, what the finder found, outcome, BIC of found vs true network
    Rate constants     true against fitted rate constants and activation energies, reaction by reaction
    1 ... 8            every run of every case: conditions, measured GC, noise-free GC, what the found network predicts,
                       and the miss in units of the GC noise (formulas)
    Search log         the search, step by step, as the finder reported it
"""
import os
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import get_column_letter as L

import make_finder_showcase as M
from smart import discovery as D

FONT = "Arial"
INK, MUTED = "1F1F1F", "5F5E5A"
HEAD_FILL = PatternFill("solid", fgColor="2F3B4C")
SUB_FILL = PatternFill("solid", fgColor="E9ECF1")
INPUT_FILL = PatternFill("solid", fgColor="FFF6CC")
GOOD = PatternFill("solid", fgColor="E2F0E4")
PART = PatternFill("solid", fgColor="FFF0D4")
BAD = PatternFill("solid", fgColor="F9E0DE")
FLAG = PatternFill(fill_type="solid", start_color="F2B8B2", end_color="F2B8B2")   # conditional formats read bgColor
THIN = Side(style="thin", color="C9CCD1")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top", wrap_text=True)
SPECIES = ["A", "B", "C", "D", "E"]
OUTCOME_FILL = {"Found exactly": GOOD, "Right reactions, a rate law differs": PART,
                "Right reactions plus an extra one": PART, "Missed a reaction": PART, "One reaction replaced by another": PART,
                "A different network": BAD, "No reaction found": BAD}

# What happened, in words, per case.  Written after reading showcase_results.json; the numbers in the
# sheet itself are all computed.  Empty = not yet written (the sheet then says so).
COMMENTS = {
    "clean": "Found the true reactions and rate laws in 93 s, on the first pass. Rate constants within 2 % of the truth, "
             "activation energies within 1 kJ/mol. The one doubt it reports itself: whether C -> D + B also runs backwards "
             "(BIC difference 5.3, below 6).",
    "few_runs": "Found the true network from only 8 runs in 44 s, rate constants within 2 %. A favourable start: on the "
                "complex network of the worked example the search found the truth in 6 of 10 such 8-run campaign starts "
                "(WORKED_EXAMPLE.md, B5). The nearest rival (A + B -> C one way only) is ruled out only just (BIC 6.9).",
    "noisy_gc": "All three true reactions found with close numbers (k within 17 %, Ea within 7 kJ/mol), plus a small extra "
                "reaction, 2 A + B -> C + D, that only explains scatter. The GC is 5x noisier than the finder assumes, so it "
                "reads noise as chemistry: BIC prefers the extra reaction by 91. Neither network fits within the GC noise "
                "and the finder says so; that warning is the cue that its noise model is wrong. Fix: measure the GC's "
                "repeatability with repeat runs. 361 s, 3 new starts.",
    "one_temperature": "Right reactions in 12 s and rate constants at 300 C within 1 %. But every activation energy comes "
                       "out as 100 +- 48 kJ/mol: the starting guess, because nothing in the data depends on temperature. "
                       "The finder warns about exactly this, and cannot say whether A + B -> C runs one way or both ways "
                       "(BIC difference 1.0). The network can be found at one temperature; its temperature dependence cannot.",
    "weak_reaction": "D never goes above 0.33 mol-%, a few times the GC noise. The finder saw that some D is made but "
                     "picked 2 C -> D + E instead of the true C -> D + B. Both fit within the GC noise and the true "
                     "network is only 4 BIC better (below 6: the data cannot tell them apart). The true network was not "
                     "among those it reported, so it did not flag the tie itself. A reaction near the detection limit can "
                     "be detected, not identified: hotter runs with longer contact time would decide.",
    "half_order": "The true rate law (order 0.5 in B) is outside what the search can build (orders 0, 1, 2). After 317 s "
                  "and 3 new starts it returned 4 reactions, one of which does not happen (A + B + C -> D + E), still "
                  "missing by 2.7x the GC noise. It does not hide this: the result says the network does not fit within "
                  "the noise. Typed in as a candidate, the true network fits (0.71x the noise, BIC 106 against 1147), so "
                  "the finder's advice (type the suspected network and rank it) works.",
    "ageing": "No network without time in it can fit: even the true reactions miss by 5.7x the GC noise. The search added "
              "a fourth reaction to soak up the drift, still missed by 5.6x, and flagged that nothing fits. Its rate "
              "constants come out 30-45 % low (an average over the ageing). It overran the 900 s limit (2653 s): the "
              "limit is checked only between steps, and this search shared the laptop with 7 others. Fix: repeat a "
              "reference run to check for ageing before trusting a network.",
    "two_routes": "Both routes to D found in 14 s; rate constants within 1 % and activation energies spot on. The routes can "
                  "be told apart because they respond differently to B: A -> D does not depend on it, C -> D + B follows "
                  "the amount of C. The varied B flows of the 30 runs make that visible.",
}


def f(size=10, bold=False, color=INK, italic=False):
    return Font(name=FONT, size=size, bold=bold, color=color, italic=italic)


def head_row(ws, row, labels, col=1):
    for j, lab in enumerate(labels):
        c = ws.cell(row=row, column=col + j, value=lab)
        c.font, c.fill, c.alignment, c.border = f(10, True, "FFFFFF"), HEAD_FILL, CENTER, BOX


def put(ws, row, col, value, fmt=None, bold=False, fill=None, align=WRAP, color=INK, italic=False):
    c = ws.cell(row=row, column=col, value=value)
    c.font, c.alignment, c.border = f(10, bold, color, italic), align, BOX
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    return c


def widths(ws, w):
    for j, x in enumerate(w, 1):
        ws.column_dimensions[L(j)].width = x


def name(wb, nm, ref):
    dn = DefinedName(nm, attr_text=ref)
    try:
        wb.defined_names[nm] = dn
    except TypeError:
        wb.defined_names.append(dn)


def rules_text(text):
    """'A + B <=> C\\n...' -> one line per reaction with its rate law (as the finder writes them)."""
    return "\n".join(D.format_network(D.parse_network(text)))


def found_text(fit):
    return "\n".join(fit.get("equations") or []) or "(no reaction)"


def data_words(c, r):
    phys = r["phys"]
    T = sorted(p[4] - 273.15 for p in phys)
    P = sorted(p[0] for p in phys)
    t = f"{T[0]:.0f} C" if abs(T[-1] - T[0]) < 0.5 else f"{T[0]:.0f}-{T[-1]:.0f} C"
    return (f"{len(phys)} runs; {t}; {P[0]:g}-{P[-1]:g} bar; GC noise {c['noise']:g}x the finder's model"
            + ("; catalyst activity 0.99 -> 0.50" if c.get("ageing") else ""))


# =========================================================================================== sheets
def read_me(wb, meta):
    ws = wb.active
    ws.title = "Read me"
    widths(ws, [3, 34, 16, 80])
    ws.sheet_view.showGridLines = False
    ws["B2"] = "Reaction finder showcase"
    ws["B2"].font = f(16, True)
    ws["B3"] = ("Theoretical data calculated from KNOWN reactions, given to the reaction finder without the answer, and what it "
                "returned, good or bad. All kinetic numbers are placeholders, like the rest of the project.")
    ws["B3"].font, ws["B3"].alignment = f(10, color=MUTED), WRAP
    ws.merge_cells("B3:D3")
    ws.row_dimensions[3].height = 30
    rows = [
        ("What this is",
         "Eight cases. Each one is a set of GC readings calculated from a network of reactions we chose, then handed to "
         "the reaction finder (smart/discovery.py, the 'Reaction finder' section of the web page) exactly as a real run "
         "log would be. The finder does not see the answer. Case 1 is the easy case; cases 2-8 each make one thing "
         "harder. The point is to show what the finder does, not to make it look good."),
        ("How the data were made",
         "The reactor model of the project: a plug-flow fixed bed at constant temperature and pressure, the mole balances "
         "integrated over the catalyst surface with the accurate solver (LSODA). Reaction rates follow the written rate "
         "laws with Arrhenius rate constants; reactions written with <=> stop at equilibrium. 30 runs spread over the "
         "operating window (2-40 bar, 5-50 mL/min of A, B and inert, 230-410 C, 1 g of catalyst), except case 2 (the 8 "
         "step-limited runs that open an optimisation campaign) and case 4 (all at 300 C). GC noise drawn from the "
         "noise model below (5 times larger in case 3). The 'answer' files next to the CSV files hold every number used."),
        ("What the finder did",
         f"Its automatic search, with the web page's time limit ({meta['max_seconds']:.0f} s): starting from no reactions "
         "it adds, removes or changes one reaction or one reaction order at a time and keeps a change only if BIC falls. "
         "Afterwards the TRUE network was fitted to the same data the same way, so that a wrong answer can be judged: did "
         "the search miss a better network, or do the data not tell the two apart?"),
        ("How to read 'Cases'",
         "Outcome compares the reactions found with the true ones. BIC (lower = better: fit to the data plus a penalty "
         "for every fitted number) is given for the found and the true network on the same data. A difference of more "
         "than 6 is a clear preference; below 6 the data cannot tell the two apart. 'Fits within GC noise' means the "
         "misses are no larger than the GC's own scatter (chi-square test at 1 %, and no species missed by more than 2x "
         "the noise on average)."),
        ("The run sheets (1-8)",
         "Every run: conditions, the GC reading the finder was given, the noise-free reading, what the found network "
         "predicts, and the miss in units of the GC noise (formulas using the noise model below). Misses above 2x the "
         "noise are shaded."),
        ("Try it yourself",
         "Web page: Reaction finder > Upload a file > choose examples/discovery/showcase/<case>.csv > Find the reactions; "
         "compare with <case>_answer.txt. Command line: python discover_reactions.py examples/discovery/showcase/clean.csv. "
         "Rebuild everything: python make_finder_showcase.py (about 45 minutes with the 8 cases in parallel, most of it case 7)."),
    ]
    r = 5
    for k, v in rows:
        ws.cell(row=r, column=2, value=k).font = f(10, True)
        ws.cell(row=r, column=2).alignment = WRAP
        c = ws.cell(row=r, column=3, value=v)
        c.font, c.alignment = f(10), WRAP
        ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 15 * max(2, len(v) // 95 + 1)
        r += 1
    r += 1
    ws.cell(row=r, column=2, value="GC noise model and thresholds (inputs used by the formulas)").font = f(11, True)
    r += 1
    head_row(ws, r, ["Input", "Value", "Meaning"], col=2)
    inputs = [("GC_REL", "Relative GC noise", 0.01, "0%", "One standard deviation of a GC reading = GC_REL x reading + GC_ABS"),
              ("GC_ABS", "Absolute GC noise (mol-%)", 0.05, "0.00", "In mol-%: 0.05 mol-% = 0.0005 as a fraction"),
              ("MISS_LIMIT", "Miss limit (x GC noise)", 2.0, "0.0", "A species is 'not explained' when its typical miss is above this"),
              ("BIC_TIE", "BIC tie (below = undecided)", 6.0, "0", "BIC difference below which two networks are not told apart (Kass & Raftery)")]
    for nm, lab, val, fmt, mean in inputs:
        r += 1
        put(ws, r, 2, lab)
        put(ws, r, 3, val, fmt=fmt, fill=INPUT_FILL, color="0000FF")
        put(ws, r, 4, mean)
        name(wb, nm, f"'Read me'!$C${r}")
    r += 2
    ws.cell(row=r, column=2, value="Colours").font = f(11, True)
    for lab, fill in (("Found the true network", GOOD), ("Partly right (one reaction or one rate law differs)", PART),
                      ("Different network, or a miss above the limit", BAD), ("Input you may change (blue text)", INPUT_FILL)):
        r += 1
        put(ws, r, 2, lab, fill=fill)
    ws.cell(row=r + 2, column=2, value=f"Made {meta['made']} by make_finder_showcase.py from showcase_results.json.").font = f(9, color=MUTED)


def versions(wb):
    ws = wb.create_sheet("Advisor versions")
    ws.sheet_view.showGridLines = False
    widths(ws, [3, 14, 40, 26, 34, 40])
    ws["B2"] = "The advisor versions in short"
    ws["B2"].font = f(14, True)
    ws["B3"] = ("All three fit a physics model (the reaction network and its rate laws) to the GC readings after every run, and a "
                "planner picks the next run within the step limits. They differ in how they correct the physics model's "
                "errors and whether they question the network. Numbers: profit earned over 25 runs, median of 10 "
                "campaigns (simulated reactor, placeholder prices).")
    ws["B3"].font, ws["B3"].alignment = f(10, color=MUTED), WRAP
    ws.merge_cells("B3:F3")
    ws.row_dimensions[3].height = 45
    head_row(ws, 5, ["Version", "How it corrects the physics model", "Reaction network", "Best at", "Weak spot"], col=2)
    rows = [
        ("v5", "One Gaussian process (GP) learns the gap between what the physics predicts and what is measured.",
         "Taken as given, never changed.", "Simple and fast; as good as the others when the chemistry is right (complex: 201).",
         "A missing reaction: one GP cannot make up for it (20)."),
        ("v6", "A committee: physics alone, physics + GP, physics + 8 neural networks, weighted by how well each predicts runs "
               "it has not seen (stacking).", "Taken as given.",
         "Best of the fixed-network versions when a reaction is missing (102).",
         "Cannot fix a wrong network; its neural networks add nothing measurable (without them: +1, p = 0.49)."),
        ("v7 (default)", "v6's committee plus one neural network alone (profit with no physics).",
         "Checked after every run; repaired when the GC readings show missing or wrong reactions.",
         "Wrong or incomplete chemistry: 200 with a reaction missing; recovers from wrong starting networks (30 of 30).",
         "Advice up to a minute slower when it searches; may keep an extra reaction; the extra neural network costs profit "
         "when no repair happens (-35)."),
    ]
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            put(ws, 6 + i, 2 + j, v, bold=(j == 0))
        ws.row_dimensions[6 + i].height = 62
    put(ws, 9, 2, "Network right", bold=True)
    put(ws, 9, 3, "All three earn the same: about 200 on the complex reaction, 442 on the simple one.")
    ws.merge_cells("C9:F9")
    r = 11
    ws.cell(row=r, column=2, value="Terms").font = f(12, True)
    terms = [
        ("Physics model", "The reaction network with rate laws k(T) P^order; its rate constants and activation energies are fitted to the GC readings."),
        ("Correction", "A model of what the physics gets wrong (GP or neural networks), added to the physics prediction."),
        ("Gaussian process (GP)", "A smooth statistical model that works well with few data points and says how sure it is."),
        ("Neural network (NN)", "A flexible learned function; needs many data points. 'Alone' = it predicts profit from the settings with no physics."),
        ("Committee / stacking", "Several models combined; each one's weight is set by how well it predicts runs it was not trained on."),
        ("Held-out error", "How badly a model predicts a run it has not seen. The weights come from it."),
        ("Network check and repair (v7)", "If the fitted network misses the GC by more than about 2x the noise, v7 searches for reactions to add, remove or replace, and keeps a change only if it clearly fits better."),
        ("Reaction finder", "The same search on its own: give it a run log and it works out which reactions explain the data (this workbook), or ranks networks you type in."),
        ("GC noise", "The expected scatter of a GC reading: 1 % of the reading + 0.05 mol-%. A miss of '1x the noise' is as close as the measurement allows."),
        ("BIC", "Bayesian information criterion: misfit plus a penalty for every fitted number; lower is better. Difference > 10 decisive, < 6 undecided."),
        ("Profit earned", "Total profit over the 25 runs of a campaign, so bad runs on the way count."),
        ("Final %", "Profit of the settings recommended after 25 runs, as % of the best possible."),
        ("Runs to 90 %", "How many runs until the recommended settings reach 90 % of the best profit."),
        ("Screening runs", "The 5 fixed opening runs every method shares, so comparisons are fair."),
        ("Step limits", "The largest change allowed between two runs (pressure, flows, temperature)."),
        ("Rules-of-thumb policy", "The scripted baseline that changes one setting at a time, like a simple manual procedure."),
        ("Paired comparison", "Same seed = same screening runs and same reactor noise for every method; p-value from the Wilcoxon signed-rank test."),
    ]
    head_row(ws, r + 1, ["Term", "Meaning"], col=2)
    for i, (t, m) in enumerate(terms):
        put(ws, r + 2 + i, 2, t, bold=True)
        put(ws, r + 2 + i, 3, m)
        ws.merge_cells(start_row=r + 2 + i, start_column=3, end_row=r + 2 + i, end_column=6)
        ws.row_dimensions[r + 2 + i].height = 30


def cases_sheet(wb, res):
    ws = wb.create_sheet("Cases")
    ws.sheet_view.showGridLines = False
    W = [4, 22, 30, 46, 24, 46, 17, 11, 11, 11, 11, 11, 24, 11, 9, 7, 64]
    widths(ws, W)
    ws["A1"] = "What the reaction finder found, case by case"
    ws["A1"].font = f(14, True)
    ws["A2"] = ("BIC columns: found and true network fitted to the same data the same way (lower = better). Columns L and M are "
                "formulas; the BIC tie threshold is on 'Read me'.")
    ws["A2"].font = f(9, color=MUTED)
    labels = ["#", "Case", "What it tests", "True reactions (rate laws)", "Data given to the finder",
              "Reactions found by the search", "Outcome", "Found network fits within GC noise?",
              "Typical miss of found network (x GC noise)", "BIC: found network", "BIC: true network",
              "BIC true minus found", "Diagnosis", "True network fits within GC noise?", "Search time (s)",
              "New starts", "What happened"]
    head_row(ws, 4, labels)
    ws.row_dimensions[4].height = 72
    for i, c in enumerate(M.CASES):
        r = res[c["key"]]
        row = 5 + i
        b, tf, fr = r["best"], r["truth_fit"], r["found_refit"]
        vals = [i + 1, c["title"], c["tests"], rules_text(c["text"]), data_words(c, r), found_text(b), r["outcome"],
                "Yes" if b["fits_noise"] else "No", b["x_noise"], fr["bic"], tf["bic"]]
        for j, v in enumerate(vals, 1):
            put(ws, row, j, v, fmt="0.00" if j == 9 else ("#,##0.0" if j in (10, 11) else None),
                bold=(j == 2), fill=OUTCOME_FILL.get(r["outcome"]) if j == 7 else None,
                align=CENTER if j in (1, 8, 9) else WRAP)
        put(ws, row, 12, f"=K{row}-J{row}", fmt="#,##0.0;-#,##0.0;0")
        put(ws, row, 13, f'=IF(G{row}="Found exactly","Found the true network",'
                         f'IF(L{row}<-BIC_TIE,"The search missed a better network (the true one)",'
                         f'IF(L{row}<=BIC_TIE,"The data cannot tell the found network from the true one",'
                         f'"The found network fits these data better than the true one")))')
        put(ws, row, 14, "Yes" if tf["fits_noise"] else "No", align=CENTER)
        put(ws, row, 15, r["seconds"], fmt="0", align=CENTER)
        put(ws, row, 16, r["restarts"], align=CENTER)
        put(ws, row, 17, COMMENTS.get(c["key"], "(see the Search log sheet)"))
        texts = {3: c["tests"], 4: rules_text(c["text"]), 5: data_words(c, r), 6: found_text(b),
                 13: "The data cannot tell the found network from the true one", 17: COMMENTS.get(c["key"], "")}
        ws.row_dimensions[row].height = 13.5 * max(est_lines(t, W[j - 1]) for j, t in texts.items()) + 6
    last = 4 + len(M.CASES)
    ws.conditional_formatting.add(f"I5:I{last}", CellIsRule(operator="greaterThan", formula=["MISS_LIMIT"], fill=FLAG))
    for col in ("H", "N"):
        ws.conditional_formatting.add(f"{col}5:{col}{last}", CellIsRule(operator="equal", formula=['"No"'], fill=FLAG))
    ws.freeze_panes = "C5"
    # tally
    t = last + 2
    ws.cell(row=t, column=2, value="Tally").font = f(11, True)
    for k, lab in enumerate(["Found exactly", "Right reactions, a rate law differs", "Right reactions plus an extra one",
                             "Missed a reaction", "One reaction replaced by another", "A different network"]):
        put(ws, t + 1 + k, 2, lab, fill=OUTCOME_FILL[lab])
        put(ws, t + 1 + k, 3, f'=COUNTIF($G$5:$G${last},B{t + 1 + k})', align=CENTER)
    return ws


def est_lines(text, width):
    """Rough number of wrapped lines of `text` in a column `width` wide (Arial 10)."""
    import math
    return sum(max(1, math.ceil(len(t) / (width * 1.05))) for t in str(text or "").splitlines() or [""])


def match(rc_list, eq):
    """The fitted constants of the reaction `eq` (canonical key without orders) in a fit, or None."""
    for rc in rc_list or []:
        if D.parse_reaction(rc["reaction"]).key().split(";")[0] == eq:
            return rc
    return None


def constants_sheet(wb, res):
    ws = wb.create_sheet("Rate constants")
    ws.sheet_view.showGridLines = False
    widths(ws, [24, 20, 30, 12, 22, 9, 10, 12, 12, 12, 10, 11, 9, 9, 9, 12, 10, 11, 9, 9])
    ws["A1"] = "True against fitted rate constants (k at 300 C per gram of catalyst) and activation energies"
    ws["A1"].font = f(13, True)
    ws["A2"] = ("Left block: the network the search found (a reaction it did not find is left empty). Right block: the true "
                "network fitted to the same data. Ratio, 'inside' and error columns are formulas. When the found network "
                "differs from the truth, its constants absorb the difference and are not comparable one to one.")
    ws["A2"].font, ws["A2"].alignment = f(9, color=MUTED), WRAP
    ws.merge_cells("A2:T2")
    ws.row_dimensions[2].height = 28
    ws.merge_cells("G3:O3")
    ws.merge_cells("P3:T3")
    for col, lab in ((7, "From the search's network"), (16, "True network fitted to the same data")):
        c = ws.cell(row=3, column=col, value=lab)
        c.font, c.fill, c.alignment = f(10, True), SUB_FILL, CENTER
    head_row(ws, 4, ["Case", "Reaction (true)", "Rate law", "True k", "k units", "True Ea (kJ/mol)",
                     "Found?", "Fitted k", "k low", "k high", "Fitted / true k", "True k inside the range?",
                     "Fitted Ea", "Ea +-", "Ea error", "Fitted k", "Fitted / true k", "Fitted Ea", "Ea +-", "Ea error"])
    ws.row_dimensions[4].height = 44
    row = 5
    for c in M.CASES:
        r = res[c["key"]]
        net = D.parse_network(c["text"])
        for j, rx in enumerate(net):
            eq = rx.key().split(";")[0]
            got = match(r["best"].get("rate_constants"), eq)
            tru = match(r["truth_fit"].get("rate_constants"), eq)
            put(ws, row, 1, c["title"], bold=(j == 0))
            put(ws, row, 2, rx.equation())
            put(ws, row, 3, rx.rate_law(j + 1))
            put(ws, row, 4, c["k"][j], fmt="0.00E+00", color="0000FF")
            put(ws, row, 5, rx.k_units())
            put(ws, row, 6, c["ea"][j] / 1000, fmt="0", color="0000FF")
            same_law = got is not None and D.parse_reaction(got["reaction"]).key() == rx.key()
            put(ws, row, 7, "Yes" if same_law else ("Other rate law" if got else "No"), align=CENTER,
                fill=GOOD if same_law else (PART if got else BAD))
            if got:
                put(ws, row, 8, got["k_ref"], fmt="0.00E+00")
                put(ws, row, 9, got["k_ref_low"], fmt="0.00E+00")
                put(ws, row, 10, got["k_ref_high"], fmt="0.00E+00")
                put(ws, row, 11, f"=H{row}/D{row}", fmt="0.00")
                put(ws, row, 12, f'=IF(AND(D{row}>=I{row},D{row}<=J{row}),"Yes","No")', align=CENTER)
                put(ws, row, 13, got["Ea_kJ"], fmt="0")
                put(ws, row, 14, got["Ea_sd_kJ"], fmt="0")
                put(ws, row, 15, f"=ROUND(M{row}-F{row},0)", fmt="+0;-0;0")
            else:
                for col in range(8, 16):
                    put(ws, row, col, None)
            if tru:
                put(ws, row, 16, tru["k_ref"], fmt="0.00E+00")
                put(ws, row, 17, f"=P{row}/D{row}", fmt="0.00")
                put(ws, row, 18, tru["Ea_kJ"], fmt="0")
                put(ws, row, 19, tru["Ea_sd_kJ"], fmt="0")
                put(ws, row, 20, f"=ROUND(R{row}-F{row},0)", fmt="+0;-0;0")
            else:
                for col in range(16, 21):
                    put(ws, row, col, None)
            row += 1
    ws.conditional_formatting.add(f"L5:L{row - 1}", CellIsRule(operator="equal", formula=['"No"'], fill=FLAG))
    ws.freeze_panes = "C5"


def run_sheet(wb, i, c, r):
    ws = wb.create_sheet(f"{i} {c['key']}"[:31])
    ws.sheet_view.showGridLines = False
    age = bool(c.get("ageing"))
    ws["A1"] = f"Case {i}: {c['title']}"
    ws["A1"].font = f(13, True)
    info = [("What it tests", c["tests"]), ("True reactions", rules_text(c["text"]).replace("\n", ";   ")),
            ("Found by the search", found_text(r["best"]).replace("\n", ";   ")),
            ("Outcome", f"{r['outcome']}; the found network {r['best']['verdict']} ({r['best']['model_words']})"),
            ("Run log file", f"examples/discovery/showcase/{c['key']}.csv (answer: {c['key']}_answer.txt)")]
    for k, (lab, v) in enumerate(info):
        ws.cell(row=2 + k, column=1, value=lab).font = f(10, True)
        ws.cell(row=2 + k, column=4, value=v).font = f(10)
    ws.cell(row=5, column=4).fill = OUTCOME_FILL.get(r["outcome"], SUB_FILL)
    cond = ["Run", "P (bar)", "Flow A (mL/min)", "Flow B (mL/min)", "Flow inert (mL/min)", "T (C)", "Catalyst (g)"]
    if age:
        cond.append("Catalyst activity")
    n0 = len(cond)
    blocks = [("GC reading given to the finder (mol-%)", SPECIES + ["I"]), ("Noise-free reading (mol-%)", SPECIES),
              ("Predicted by the found network (mol-%)", SPECIES), ("Miss, x GC noise (|predicted - given| / noise)", SPECIES)]
    hr = 9
    col = n0 + 1
    starts = {}
    for title, sp in blocks:
        ws.merge_cells(start_row=hr - 1, start_column=col, end_row=hr - 1, end_column=col + len(sp) - 1)
        cc = ws.cell(row=hr - 1, column=col, value=title)
        cc.font, cc.fill, cc.alignment = f(10, True), SUB_FILL, CENTER
        starts[title[:5]] = col
        col += len(sp)
    worst_col = col
    head_row(ws, hr, cond + [s for _, sp in blocks for s in sp] + ["Worst miss"])
    ws.row_dimensions[hr].height = 32
    phys, gc, gt, pr = r["phys"], r["gc"], r["gc_true"], r["best"]["predicted_gc"]
    gcol, tcol, pcol, mcol = starts["GC re"], starts["Noise"], starts["Predi"], starts["Miss,"]
    first = hr + 1
    for k in range(len(phys)):
        row = first + k
        p = phys[k]
        vals = [k + 1, p[0], p[1], p[2], p[3], p[4] - 273.15, p[5]] + ([r["activity"][k]] if age else [])
        fmts = [None, "0.0", "0.0", "0.0", "0.0", "0", "0.0", "0.00"]
        for j, v in enumerate(vals):
            put(ws, row, 1 + j, v, fmt=fmts[j], align=CENTER)
        for s in range(6):
            put(ws, row, gcol + s, 100 * gc[k][s], fmt="0.000")
        for s in range(5):
            put(ws, row, tcol + s, 100 * gt[k][s], fmt="0.000", color=MUTED)
            put(ws, row, pcol + s, 100 * pr[k][s], fmt="0.000")
            g, q = L(gcol + s), L(pcol + s)
            put(ws, row, mcol + s, f"=ABS({q}{row}-{g}{row})/(GC_REL*{g}{row}+GC_ABS)", fmt="0.0")
        put(ws, row, worst_col, f"=MAX({L(mcol)}{row}:{L(mcol + 4)}{row})", fmt="0.0", bold=True)
    last = first + len(phys) - 1
    ws.conditional_formatting.add(f"{L(mcol)}{first}:{L(worst_col)}{last}",
                                  CellIsRule(operator="greaterThan", formula=["MISS_LIMIT"], fill=FLAG))
    # summary rows
    s1, s2 = last + 1, last + 2
    put(ws, s1, mcol - 1, "Typical miss (root mean square)", bold=True, align=Alignment(horizontal="right", wrap_text=True))
    put(ws, s2, mcol - 1, "Largest miss", bold=True, align=Alignment(horizontal="right", wrap_text=True))
    for s in range(5):
        cl = L(mcol + s)
        put(ws, s1, mcol + s, f"=SQRT(SUMSQ({cl}{first}:{cl}{last})/COUNT({cl}{first}:{cl}{last}))", fmt="0.00", bold=True)
        put(ws, s2, mcol + s, f"=MAX({cl}{first}:{cl}{last})", fmt="0.0")
    ws.conditional_formatting.add(f"{L(mcol)}{s1}:{L(mcol + 4)}{s1}",
                                  CellIsRule(operator="greaterThan", formula=["MISS_LIMIT"], fill=FLAG))
    ws.cell(row=s2 + 2, column=1, value=("The miss uses the GC reading given to the finder as the reference, as the finder "
                                         "does. 'Typical miss' matches the finder's own per-species number (it fits with a "
                                         "slightly wider error allowance, so its BIC is not computed from these cells).")
            ).font = f(9, color=MUTED, italic=True)
    widths(ws, [6, 8, 9, 9, 9, 7, 9] + ([9] if age else []) + [8] * (6 + 5 + 5 + 5) + [9])
    ws.column_dimensions["A"].width = 6
    ws.freeze_panes = ws.cell(row=first, column=n0 + 1)


def log_sheet(wb, res):
    ws = wb.create_sheet("Search log")
    ws.sheet_view.showGridLines = False
    widths(ws, [26, 8, 150])
    ws["A1"] = "The search, step by step, as the finder reported it"
    ws["A1"].font = f(13, True)
    head_row(ws, 3, ["Case", "Line", "Text"])
    row = 4
    for c in M.CASES:
        r = res[c["key"]]
        lines = [("path", x) for x in r["path"]] + [("summary", x) for x in r["summary"]]
        for k, (_, x) in enumerate(lines, 1):
            put(ws, row, 1, c["title"] if k == 1 else None, bold=True)
            put(ws, row, 2, k, align=CENTER)
            put(ws, row, 3, x)
            ws.row_dimensions[row].height = 13 * max(1, len(x) // 150 + 1)
            row += 1
        row += 1
    ws.freeze_panes = "A4"


def build(path=None, src=None):
    src = src or os.path.join(M.OUT, "showcase_results.json")
    with open(src, encoding="utf-8") as fh:
        meta = json.load(fh)
    res = meta["results"]
    for c in M.CASES:                                     # recomputed, so the sheet follows the current definition
        res[c["key"]]["outcome"] = M.outcome(c["text"], res[c["key"]]["best"]["reactions"])
    wb = Workbook()
    read_me(wb, meta)
    versions(wb)
    cases_sheet(wb, res)
    constants_sheet(wb, res)
    for i, c in enumerate(M.CASES, 1):
        run_sheet(wb, i, c, res[c["key"]])
    log_sheet(wb, res)
    wb.active = wb.sheetnames.index("Cases")
    out = path or os.path.join(M.OUT, "finder_showcase.xlsx")
    wb.save(out)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build()
