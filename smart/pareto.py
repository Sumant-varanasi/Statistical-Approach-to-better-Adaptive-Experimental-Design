"""
The profit versus selectivity trade-off curve (a Pareto front), as an OPTIONAL extra.

The advisor normally gives one answer: the settings it expects to earn the most, if wanted
subject to a minimum selectivity.  Often an operator would give up a little profit for a purer
product (less by-product to separate, a customer specification).  This module shows every
sensible compromise so that the operator can pick one.  Nothing here runs unless it is asked
for, it changes nothing in the advisor, and it draws no random numbers from the advisor (it
has its own random generator), so the advisor's next suggestion is the same with or without it.

Words used
----------
selectivity    of all the A that reacted, the share that ended up as the wanted product C
               (the rest became by-products).  1.0 = no by-product at all.
trade-off      settings that earn the most usually push the reactor hard (hot, long contact
               time), which also makes more by-product; gentler settings make a purer product
               but less of it per hour.
the curve      the settings that are not beaten on both counts at once: a setting is left out
(Pareto front) when some other setting is predicted to give BOTH more profit AND a higher
               selectivity.  What is left runs from "most profitable" to "most selective that
               still makes a profit"; along it profit can only fall as selectivity rises.
balanced point the knee of the curve: up to it extra selectivity is cheaper than beyond it (less
               profit given up per point of selectivity).  Found as the point farthest above
               the straight line that joins the most profitable point and the most selective
               point that still makes a profit, both axes scaled to 0..1.  A curve that is
               straight or bends the other way has no such point: none is given then (the
               'balanced' marker falls on the most profitable point and the text says so).

What is predicted
-----------------
profit         the advisor's own prediction (the fitted chemistry plus its correction for what
               the physics misses), exactly what it plans with; range = +- 1.64 predictive sd
               (9 out of 10 times the truth should fall inside, the convention of the web page)
selectivity,   from the fitted chemistry alone (the correction acts on profit only), with a
conversion     range from the spread between the advisor's plausible versions of the chemistry

How the curve is found
----------------------
1. candidates over the whole allowed box (Sobol points, the runs already done, points around
   the best run), with the fixed knobs pinned and the minimum total flow respected
2. keep the non-dominated ones; then refine: take points spread along the current curve (a
   ladder of selectivity levels, each holding the most profitable setting found at that
   selectivity or better), try small changes of the knobs around each and half-way points
   between neighbours, and keep what improves the curve; repeat with smaller changes
3. cut the curve where profit reaches zero, thin it to about n_points evenly spread along
   its length, and predict those few points again with every plausible chemistry
The search uses the advisor's best-estimate chemistry only (one reactor simulation per
candidate instead of one per plausible chemistry), which is what keeps it to a few seconds.

    from smart.pareto import tradeoff, spec_for_point
    res = tradeoff(adv)                         # adv: a SmartAdvisor after adv.update(...)
    res["points"][res["markers"]["balanced"]]   # settings, profit, selectivity, ranges, ...
    adv.spec = spec_for_point(res["points"][3], adv)    # "aim for this point" = a minimum selectivity

true_tradeoff / true_values / compare_with_truth work on the virtual reactor (noise-free, fresh
catalyst) and are for scoring only: the advisor never sees them.

Measured (test_pareto.py: complex virtual reactor, 12 runs, seed 0, v5 advisor): tradeoff() takes
about 2 s on a quiet machine (7 to 8 s were measured while other heavy jobs were running; the
target is 10 s).  Over the points predicted to earn at least 80 % of the top profit, the settings on
the predicted curve truly lie within 0.33 % (of the best possible profit) of the true curve and
the predicted selectivity is within 0.07 percentage point, but the predicted profit is 0.4 to
0.65 per hour (2 to 3.5 %) too high all along the curve: the shape and the settings are right,
the level carries the noise of the few profit measurements.  The selective end (profit near
zero) is the weakest part: its last point is 4.6 % below the true curve.  A second campaign
(seed 1, checked by hand, not in the test) was less even: predicted profit 0.2 to 0.9 per hour
too high, the error growing toward the selective end (with the v6 committee correction -0.2 to
+1.4), and the last point 5.3 % below the true curve; its profitable end was again within
0.5 per hour and 0.2 % of the true curve (v5).  Two campaigns: read these as examples, not as
a guarantee.

Known limits (measured on the seed 0 campaign)
* The curve is found by a random search, not by an exact optimiser.  A 6 times longer search
  of the same model finds settings 0.03 to 0.05 per hour better on average and up to 0.2 better
  at single points; another search seed moves the curve by about 0.1 per hour on average and
  by up to 0.7 on the steep selective end.
* 'Aim for this point' hands the advisor a minimum selectivity (spec_for_point).  The advisor
  accepts the point, but its own planner (SmartAdvisor.plan) searches coarsely under such a
  limit: for the balanced point (14.4 per hour on the curve) it found 10.5 to 13.9 per hour
  over five random seeds.  Until the planner is given the chosen point as a starting point,
  show the point's own settings as the target.
* Other specs on the advisor (minimum conversion, maximum adiabatic rise) are judged with the
  best-estimate chemistry during the search; a returned point that the advisor's own rule
  would reject is dropped at the end, so the curve can come back with fewer points.

ALL NUMBERS ARE PLACEHOLDERS, like the rest of the twin.
"""
import os
import json
import time
import hashlib
import numpy as np
from .systems import SYSTEMS, LOW, HIGH, MIN_TOTAL_FLOW, WCAT, A_SPEC, DIM, KEYS, unscale, feasible
from .advisor import SPEC_KEYS, sobol_in, kpis_from_gc

Z90 = 1.64                       # +- this many sd = the 90 % range shown on the web page
ROUND_SD0, ROUND_SHRINK = 0.08, 0.75   # size of the knob changes tried in refinement round r: 0.08 * 0.75^r (scaled units)
N_ANCHOR, N_LOCAL = 16, 32       # refinement: points along the curve, trial changes around each
TRUE_ROUNDS, TRUE_ANCHOR, TRUE_LOCAL = 16, 24, 48     # the true curve is refined much further
_BLOCK = 256                     # runs solved together by the exact solver
NO_ANSWER = -900.0               # the advisor marks "the physics gave no answer here" with -1e3
DEC_P, DEC_S = 3, 4              # decimals of profit and of selectivity / conversion in the results
KNEE_MIN = 0.05                  # a balanced point must stand this far above the line joining the ends (axes scaled 0..1)
SEL_RES = 0.002                  # a curve spanning less selectivity than this (0.2 percentage point) is no trade-off:
#                                  the noise of one GC reading alone moves the selectivity read off it by 0.1 to 0.35 point


# =================================================================================== small tools
def pareto_front(profit, selectivity, tol=1e-9):
    """Indices of the non-dominated points, ordered by rising selectivity (so by falling
    profit).  A point is dropped when another one has at least its profit and at least its
    selectivity; what is left is the trade-off curve."""
    profit, sel = np.asarray(profit, float), np.asarray(selectivity, float)
    keep, best = [], -np.inf
    for i in np.lexsort((-sel, -profit)):          # most profitable first; ties: most selective first
        if sel[i] > best + tol:
            keep.append(int(i))
            best = sel[i]
    return np.array(keep, int)


def _shown_front(profit, sel):
    """The curve as it will be shown: non-dominated in the ROUNDED numbers, so that two points
    never show the same profit (or the same selectivity) after rounding."""
    return pareto_front(np.round(profit, DEC_P), np.round(sel, DEC_S))


def _unit(v):
    v = np.asarray(v, float)
    return (v - v.min()) / max(float(np.ptp(v)), 1e-12)


def _arc(profit, sel):
    """Distance travelled along the curve, both axes scaled to 0..1."""
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(_unit(profit)), np.diff(_unit(sel))))]


def _spread(profit, sel, n):
    """Positions of about n points evenly spread along the curve, both ends included."""
    m = len(profit)
    if m <= max(n, 2):
        return np.arange(m)
    arc = _arc(profit, sel)
    return np.unique([int(np.argmin(np.abs(arc - a))) for a in np.linspace(0.0, arc[-1], n)])


def _knee(profit, sel):
    """Position of the balanced point: farthest above the straight line joining the ends, both
    axes scaled to 0..1.  Above that line the profit given up per point of selectivity is
    smaller before the point than after it, which is what 'balanced' promises.  0 (= the most
    profitable end, 'no balanced point') when the curve has fewer than 3 points or no point
    stands out by at least KNEE_MIN: a straight or inward-bent curve has no knee, and picking
    the largest of its rounding wiggles would promise a bargain that is not there."""
    if len(profit) < 3:
        return 0
    lift = _unit(profit) + _unit(sel) - 1.0
    k = int(np.argmax(lift))
    return k if lift[k] >= KNEE_MIN else 0


def _n_profitable(profit):
    """How many points of a curve (ordered by falling profit) make a profit, as shown after
    rounding; the whole curve when not even its best point does."""
    profit = np.round(np.asarray(profit, float), DEC_P)
    return int(np.sum(profit >= 0.0)) if profit[0] > 0 else len(profit)


def _floor(p_max, min_profit):
    """Lowest profit still shown.  Normally min_profit (0 = break-even).  When nothing is
    predicted to reach it: break-even if something at least makes a profit, otherwise the
    least-loss compromises (down to about one money unit per hour below the best)."""
    if min_profit is None:
        return -np.inf
    if p_max > min_profit:
        return float(min_profit)
    return 0.0 if p_max > 0 else float(p_max - (abs(p_max) + 1.0))


def _num(v, d=4):
    """A plain Python number for JSON; not-a-number becomes None."""
    if v is None:
        return None
    v = float(v)
    return float(np.round(v, d)) if np.isfinite(v) else None


def _settings(x):
    """Scaled knobs -> the five set points in physical units (T in K, and T_C in degrees C)."""
    ph = unscale(x)[0]
    out = {k: _num(v, 3) for k, v in zip(KEYS, ph)}
    out["T_C"] = _num(ph[4] - 273.15, 2)
    return out


def _say(x):
    P, VA, VB, VI, T = unscale(x)[0]
    return f"{P:.0f} bar, A {VA:.0f}, B {VB:.0f}, inert {VI:.0f} mL/min, {T - 273.15:.0f} °C"


# =================================================================================== the search
def _search(evalf, pin, lo, hi, free, seeds, n_candidates, rng, min_profit, rounds, n_anchor=N_ANCHOR,
            n_local=N_LOCAL):
    """Candidates -> non-dominated set -> refinement around points spread along it.
    evalf(C) -> profit, selectivity, conversion, usable (one value per row of C).
    Returns every usable point tried: C, profit, selectivity, conversion."""
    pool = [np.empty((0, DIM)), np.empty(0), np.empty(0), np.empty(0)]

    def add(C):
        C = pin(np.atleast_2d(np.asarray(C, float)).copy())
        C = C[feasible(unscale(C))]
        if not len(C):
            return
        p, s, xa, ok = evalf(C)
        ok = ok & np.isfinite(p) & np.isfinite(s) & (p > NO_ANSWER)
        for k, v in enumerate((C, p, s, xa)):
            pool[k] = np.concatenate([pool[k], v[ok]])

    parts = [sobol_in(lo, hi, int(n_candidates), rng, free)]
    if seeds is not None and len(seeds):
        parts.append(seeds)
    add(np.vstack(parts))
    for r in range(max(int(rounds), 0)):
        sd = ROUND_SD0 * ROUND_SHRINK ** r
        C, p, s, _ = pool
        if not len(p):
            break
        F = pareto_front(p, s)
        p_max = p[F[0]]
        fl = _floor(p_max, min_profit)
        seg = F[p[F] >= fl - 0.25 * (p_max - fl)]          # a margin below the floor, so the far end is refined too
        if len(seg) < 4:
            seg = F[:4]
        A = C[seg[_spread(p[seg], s[seg], n_anchor)]]
        noise = rng.normal(0.0, sd, (len(A), n_local, DIM))
        some = rng.random((len(A), n_local, DIM)) < 0.5     # half of the trials change only some of the knobs
        some[:, :n_local // 2] = True
        trial = (A[:, None, :] + noise * some).reshape(-1, DIM)
        add(np.vstack([trial, 0.5 * (A[:-1] + A[1:])]) if len(A) > 1 else trial)
    return pool


def _pick(p, s, n_points, min_profit):
    """From all points tried: the curve (positions into the pool, by rising selectivity), cut
    at the profit floor; the positions of about n_points of them, and of the balanced point.
    The balanced point is looked for on the stretch that makes a profit, and the last point of
    that stretch (the most selective one that still makes a profit) is always kept.  A curve
    that spans less than SEL_RES of selectivity is not a trade-off: only its most profitable
    point is kept."""
    F = _shown_front(p, s)
    fl = _floor(p[F[0]], min_profit)
    F = F[p[F] >= fl]
    if s[F[-1]] - s[F[0]] < SEL_RES:
        F = F[:1]
    m = _n_profitable(p[F])
    knee = _knee(p[F[:m]], s[F[:m]])
    must = {0, len(F) - 1, knee, m - 1}
    sel = set(int(i) for i in _spread(p[F], s[F], max(int(n_points), 2)))
    for i in sorted(must - sel):                            # make room for the marked points
        spare = [j for j in sel if j not in must]
        if spare:
            sel.remove(min(spare, key=lambda j: abs(j - i)))
        sel.add(i)
    return F, np.array(sorted(sel), int), knee, fl


def _box(lo, hi, fixed):
    lo = np.zeros(DIM) if lo is None else np.asarray(lo, float)
    hi = np.ones(DIM) if hi is None else np.asarray(hi, float)
    fixed = dict(fixed or {})

    def pin(C):
        C = np.clip(C, lo, hi)
        for i, v in fixed.items():
            C[:, i] = v
        return C
    return lo, hi, fixed, [i for i in range(DIM) if i not in fixed], pin


def _markers(points, knee_pos, sel_pos):
    """Label the three points an operator will ask about.  knee_pos: the balanced point's
    position on the full curve; sel_pos: the positions of the returned points on it.
    most_selective is the most selective point that still makes a profit (the last point of the
    curve unless it was asked to continue below break-even); when nothing makes a profit it is
    the last point.  balanced lies on the stretch up to it, and equals most_profitable when
    the curve has no knee."""
    P = np.array([pt["profit"] for pt in points], float)
    S = np.array([pt["selectivity"] for pt in points], float)
    m = _n_profitable(P)
    hit = np.nonzero(np.asarray(sel_pos) == knee_pos)[0]
    bal = int(hit[0]) if len(hit) and hit[0] < m else _knee(P[:m], S[:m])
    mk = dict(most_profitable=0, balanced=bal, most_selective=m - 1)
    names = dict(most_profitable="most profitable", balanced="balanced", most_selective="most selective")
    for pt in points:
        pt["label"] = ""
    for key in ("most_selective", "balanced", "most_profitable"):      # when two coincide, the later name here is shown
        points[mk[key]]["label"] = names[key]
    return mk


def _curve(F, p, s, n=60):
    j = F[_spread(p[F], s[F], n)]
    return dict(selectivity=[_num(v, DEC_S) for v in s[j]], profit=[_num(v, DEC_P) for v in p[j]])


# =================================================================================== the advisor's curve
def tradeoff(adv, n_points=12, n_candidates=1024, seed=0, rng=None, min_profit=0.0, near_steps=2.0,
             rounds=8, respect_spec=True):
    """The predicted profit versus selectivity trade-off curve of a fitted SmartAdvisor.

    adv           a SmartAdvisor after adv.update(...).  It is only read: its box (lo, hi), fixed
                  knobs, step limits, plausible chemistries and correction are used as they are.
    n_points      about how many points to return, evenly spread along the curve
    n_candidates  Sobol points over the allowed box that start the search
    seed / rng    the search's own random generator (never the advisor's)
    min_profit    the curve is cut where the predicted profit falls below this (0 = break-even);
                  None = keep the whole curve, however much the selective end loses.  If no
                  setting reaches it the curve is cut at break-even instead, and if nothing makes
                  a profit at all the least-loss compromises are shown; the explanation says which.
    near_steps    a point farther than this many allowed steps from every run done so far is
                  flagged as an extrapolation
    rounds        refinement rounds (each tries smaller changes of the knobs than the one before)
    respect_spec  keep the advisor's other specs (minimum conversion, maximum adiabatic rise) as
                  limits, judged with the best-estimate chemistry during the search; a point the
                  advisor's own rule (at least half of its plausible chemistries meet them) would
                  reject is dropped at the end.  A minimum-selectivity spec is NOT applied (the
                  curve is there to show what other choices of it would give); every point says
                  whether it meets the present one.

    Returns a dict that json.dumps accepts:
      points      list, by rising selectivity (so falling profit), each a dict with
                    P, V_A, V_B, V_I, T (bar, mL/min, K), T_C (degrees C), x (scaled knobs)
                    profit, profit_lo, profit_hi, profit_sd      predicted, with its 90 % range
                    selectivity, selectivity_lo, selectivity_hi  fraction 0..1
                    conversion, conversion_lo, conversion_hi     fraction 0..1
                    steps_away               allowed steps from the last run to get there
                    steps_from_nearest_run, nearest_run (1 = first run), near_data, extrapolation
                    min_selectivity_spec     the minimum selectivity to set to aim here: the largest
                                             limit under which the advisor still accepts this point
                    meets_current_spec       against the advisor's present minimum selectivity (None if none)
                    label                    'most profitable' / 'balanced' / 'most selective' / ''
      markers     positions in points: most_profitable (always 0), most_selective (the most
                  selective point that still makes a profit: the last point, unless min_profit
                  let the curve continue below break-even; the last point too when nothing makes
                  a profit) and balanced (between the two; equal to most_profitable when the
                  curve has no clear knee, see _knee)
      curve       selectivity and profit of up to 60 points, for drawing the line
      reference   best_run (the run the advisor recommends) and last_run, measured and predicted
      explanation plain-word sentences for someone who has never seen such a curve
      wcat, a_spec  the catalyst load the prediction is for (the advisor's)
      spec, system, product, n_runs, n_evaluated, seconds, settings, units

    A single point comes back when the allowed settings offer no real choice (all knobs fixed,
    or the whole curve spans less than 0.2 percentage point of selectivity, which one GC reading
    cannot tell apart).
    """
    if getattr(adv, "n", 0) < 1 or not hasattr(adv, "thetas"):
        raise ValueError("the advisor has no runs yet: call adv.update(...) before asking for the trade-off curve")
    t0 = time.time()
    rng = np.random.default_rng(seed) if rng is None else rng
    step = np.where(adv.step > 0, adv.step, 1.0)
    other = {k: v for k, v in adv.spec.items() if k != "S_C"} if respect_spec else {}

    def within(XA, SC, dT):
        ok = np.ones(np.shape(SC), bool)
        for key, lim in other.items():
            val = {"X_A": XA, "dT_ad": dT}[key]
            with np.errstate(invalid="ignore"):
                ok &= (val >= lim) if SPEC_KEYS[key][0] == "min" else (val <= lim)
        return ok

    def evalf(C):                                   # best-estimate chemistry + the advisor's correction
        pr, XA, SC, dT = adv._physics(C, adv.thetas[:1], kpis=True)
        mu_p = np.nan_to_num(pr[0], nan=-1e3)
        mu = mu_p + adv._correct(C, mu_p)[0]
        return mu, SC[0], XA[0], np.isfinite(pr[0]) & within(XA[0], SC[0], dT[0])

    x_best = adv.X[int(np.argmax(adv.mu_data))]
    seeds = np.vstack([adv.X, x_best + rng.normal(0.0, 0.5, (64, DIM)) * adv.step])
    C, p, s, _ = _search(evalf, adv._pin, adv.lo, adv.hi, adv.free, seeds, n_candidates, rng, min_profit, rounds)
    if not len(p):
        raise ValueError("no setting inside the allowed ranges gives a usable prediction; widen the ranges "
                         "or relax the product spec")
    F, sel_pos, knee_pos, fl = _pick(p, s, n_points, min_profit)
    Xf = C[F[sel_pos]]

    # ---- the few points that are returned: predicted again with every plausible chemistry
    mu, sd = adv.predict(Xf)
    pr, XA, SC, dT = adv._physics(Xf, adv.thetas, kpis=True)
    with np.errstate(invalid="ignore"):
        s_sd, x_sd = np.nan_to_num(np.nanstd(SC, 0)), np.nan_to_num(np.nanstd(XA, 0))
        # the advisor accepts a setting under a minimum-selectivity spec when at least half of the
        # plausible chemistries meet it: the largest limit this point passes is the upper median
        valid = np.isfinite(pr) & np.isfinite(SC) & within(XA, SC, dT)
        ranked = -np.sort(-np.where(valid, SC, -np.inf), axis=0)
    s_spec = np.floor(ranked[int(np.ceil(len(adv.thetas) / 2)) - 1] * 1e4) / 1e4
    # the search judged the other specs with the best-estimate chemistry only; a point that the
    # advisor's own rule (at least half of the plausible chemistries meet them) rejects is left out
    accept = valid.mean(0) >= 0.5
    cand = np.nonzero(accept)[0] if other and accept.any() else np.arange(len(mu))
    keep = cand[_shown_front(mu[cand], SC[0][cand])]   # the full prediction has the last word
    keep = keep[mu[keep] >= fl - 1e-9] if (mu[keep] >= fl - 1e-9).any() else keep[:1]
    d_run = np.max(np.abs(Xf[:, None, :] - adv.X[None, :, :]) / step, axis=2)      # (points, runs), in steps
    cur = adv.spec.get("S_C")
    points = []
    for i in keep:
        j = int(np.argmin(d_run[i]))
        near = bool(d_run[i, j] <= near_steps + 1e-9)
        points.append(dict(
            **_settings(Xf[i]), x=[float(v) for v in Xf[i]],
            profit=_num(mu[i], DEC_P), profit_lo=_num(mu[i] - Z90 * sd[i], DEC_P),
            profit_hi=_num(mu[i] + Z90 * sd[i], DEC_P), profit_sd=_num(sd[i], DEC_P),
            selectivity=_num(SC[0, i], DEC_S), selectivity_lo=_num(np.clip(SC[0, i] - Z90 * s_sd[i], 0, 1), DEC_S),
            selectivity_hi=_num(np.clip(SC[0, i] + Z90 * s_sd[i], 0, 1), DEC_S),
            conversion=_num(XA[0, i], DEC_S), conversion_lo=_num(np.clip(XA[0, i] - Z90 * x_sd[i], 0, 1), DEC_S),
            conversion_hi=_num(np.clip(XA[0, i] + Z90 * x_sd[i], 0, 1), DEC_S),
            steps_away=int(adv.steps_away(adv.X[-1], Xf[i])),
            steps_from_nearest_run=_num(d_run[i, j], 2), nearest_run=j + 1, near_data=near, extrapolation=not near,
            min_selectivity_spec=_num(s_spec[i]),
            meets_current_spec=None if cur is None else bool(np.isfinite(s_spec[i]) and s_spec[i] >= cur - 1e-9)))
    markers = _markers(points, knee_pos, sel_pos[keep])

    # ---- the operator's own runs, for reference
    ph6 = np.column_stack([unscale(adv.X), adv.W])
    XAm, SCm, _ = kpis_from_gc(adv.S, ph6, adv.gc)

    def run_row(i):
        return dict(run=int(i) + 1, **_settings(adv.X[i]), x=[float(v) for v in adv.X[i]],
                    profit_measured=_num(adv.y[i], 3), profit_predicted=_num(adv.mu_data[i], 3),
                    selectivity=_num(SCm[i]), conversion=_num(XAm[i]))

    reference = dict(best_run=run_row(adv.recommend()), last_run=run_row(adv.n - 1))
    return dict(
        ok=True, system=adv.S.name, product=adv.S.species[adv.S.iC], n_runs=int(adv.n),
        wcat=float(adv.wcat), a_spec=float(adv.a_spec),
        points=points, markers=markers, curve=_curve(F, p, s), reference=reference,
        spec={k: float(v) for k, v in adv.spec.items()},
        explanation=_explain(points, markers, near_steps, min_profit),
        n_evaluated=int(len(p)), seconds=_num(time.time() - t0, 2),
        settings=dict(n_points=int(n_points), n_candidates=int(n_candidates), seed=None if seed is None else int(seed),
                      min_profit=None if min_profit is None else float(min_profit), near_steps=float(near_steps),
                      rounds=int(rounds), respect_spec=bool(respect_spec)),
        units=dict(P="bar", V_A="mL/min", V_B="mL/min", V_I="mL/min", T="K", T_C="°C",
                   profit="money units per hour (placeholder prices)", selectivity="fraction, 0 to 1",
                   conversion="fraction, 0 to 1"))


def _explain(points, mk, near_steps, min_profit):
    """A few plain sentences about the curve, for someone who has never seen one."""
    a, b, c = (points[mk[k]] for k in ("most_profitable", "balanced", "most_selective"))
    z = points[-1]                                                    # the end of the curve (c unless it goes on at a loss)
    pct = lambda v: f"{100 * v:.1f} %"
    num = lambda v: f"{v:.2f}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}")
    where = lambda pt: _say(np.array(pt["x"]))
    top = a["profit"]
    no_profit = top <= 0                                              # the curve shows the least-loss compromises
    not_reached = min_profit is not None and 0 < top <= min_profit    # the asked-for profit is out of reach: cut at break-even
    still = "" if no_profit else " that still make a profit"          # what the most selective marked point promises
    if no_profit or min_profit is None or z["profit"] < 0:
        end = "."
    elif min_profit > 0 and not not_reached:
        end = f" that still earns at least {min_profit:g} per hour."
        still = f" that still earn at least {min_profit:g} per hour"
    else:
        end = " that still makes a profit."
    out = ["Each point on the curve is a setting where the product cannot be made purer without earning less. "
           "Selectivity is the share of the reacted feed that ends up as the wanted product instead of by-products; "
           "one end of the curve earns the most, the other end makes the purest product" + end]
    if no_profit:
        out.append("None of the allowed settings is predicted to make a profit, so the curve shows the compromises "
                   "that lose the least.")
    elif not_reached:
        out.append(f"None of the allowed settings is predicted to earn the {min_profit:g} per hour asked for (the best "
                   f"is {top:.1f}), so the curve is shown down to break-even instead.")
    lead = "the settings that lose the least" if no_profit else "the most profitable settings"

    def fall(frm, to):
        d = frm["profit"] - to["profit"]
        if frm["profit"] > 0 and to["profit"] >= 0:
            share = 100 * d / frm["profit"]
            return f"profit falls by {share:.{0 if share >= 10 else 1 if share >= 1 else 2}f} %" \
                   f" (from {frm['profit']:.1f} to {to['profit']:.1f} per hour)"
        return f"profit falls by {d:.1f} per hour (from {frm['profit']:.1f} to {to['profit']:.1f})"

    def cost(frm, to):                                                # profit given up per percentage point of selectivity
        return num((frm["profit"] - to["profit"]) / max(100 * (to["selectivity"] - frm["selectivity"]), 1e-9))

    if len(points) == 1:
        out.append(f"Here there is no real trade-off to choose from inside your allowed ranges: {lead} ({where(a)}) "
                   f"give {pct(a['selectivity'])} selectivity at {a['profit']:.1f} per hour, and no other setting on the "
                   f"curve is more selective by as much as {100 * SEL_RES:g} percentage point (less than one GC "
                   f"reading can tell apart).")
    elif c is a:
        out.append(f"Only the most profitable settings ({where(a)}) are predicted to make a profit: "
                   f"{pct(a['selectivity'])} selectivity at {a['profit']:.1f} per hour.")
    elif mk["balanced"] not in (mk["most_profitable"], mk["most_selective"]):
        out.append(f"Moving from {lead} ({where(a)}) to the balanced point ({where(b)}), "
                   f"selectivity rises from {pct(a['selectivity'])} to {pct(b['selectivity'])} and {fall(a, b)}. "
                   f"Beyond the balanced point purity costs more: about {cost(b, c)} per hour for every further "
                   f"percentage point of selectivity, against {cost(a, b)} up to it. The most selective settings{still} "
                   f"({where(c)}) reach {pct(c['selectivity'])} at {c['profit']:.1f} per hour.")
    else:
        out.append(f"Moving from {lead} ({where(a)}) to the most selective ones{still} ({where(c)}), "
                   f"selectivity rises from {pct(a['selectivity'])} to {pct(c['selectivity'])} and {fall(a, c)}: about "
                   f"{cost(a, c)} per hour for every percentage point of selectivity. The curve has no clear knee (no "
                   f"point up to which extra purity is markedly cheaper than beyond it), so no balanced point is "
                   f"singled out.")
    if z is not c:
        out[-1] += (f" Past break-even the curve goes on to {pct(z['selectivity'])} selectivity, at a loss of "
                    f"{-z['profit']:.1f} per hour ({where(z)}).")
    far = sum(pt["extrapolation"] for pt in points)
    if far:
        out.append(f"{far} of the {len(points)} points are more than {near_steps:g} allowed steps away from every run "
                   f"done so far: there the advisor is extrapolating, so read those numbers as a guide and approach "
                   f"them one step at a time.")
    else:
        out.append(f"Every point is within {near_steps:g} allowed steps of a run already done, so the advisor has "
                   f"data nearby for all of them.")
    out.append("To aim for a point, set the minimum selectivity to the value shown with it: the advisor then steers "
               "toward the most profitable settings it can find that keep at least that selectivity. Its own search "
               "under such a limit is rougher than this curve, so it may settle for less profit than the point shows; "
               "the settings listed with the point are the target to compare its advice with.")
    return out


def spec_for_point(point, adv=None, spec=None):
    """'Aim for this point' as the advisor's existing product spec: a minimum selectivity.

    point   one entry of tradeoff(...)["points"]
    adv     the advisor (its other specs, e.g. a maximum adiabatic rise, are kept), or
    spec    a spec dict to extend instead
    Returns a new spec dict, e.g. {"S_C": 0.9312}; use it as SmartAdvisor(spec=...) or
    adv.spec = ..., then plan / suggest as usual.  The limit is the largest one under which the
    advisor still accepts this point (at least half of its plausible chemistries meet it),
    so the point itself stays allowed and the advisor looks for the best profit at that purity.
    Its planner searches coarsely under such a limit and does not know the point, so it may
    head for settings that earn less than the point (see 'Known limits' at the top)."""
    out = dict(spec if spec is not None else (adv.spec if adv is not None else {}))
    lim = point.get("min_selectivity_spec")
    if lim is None:
        lim = point["selectivity"]
    out["S_C"] = float(lim)
    return out


# =================================================================================== the true curve (scoring only)
_TRUE_FILE = os.path.join(os.path.dirname(__file__), "pareto_true.json")
_TRUE_MEMO = {}


def _system(system):
    return SYSTEMS[system] if isinstance(system, str) else system


def true_values(system, X, wcat=WCAT, a_spec=A_SPEC, exact=True):
    """Noise-free profit, selectivity and conversion of the virtual reactor with a fresh
    catalyst at scaled settings X (n x 5).  exact=True uses the tight LSODA solver (the one the
    virtual reactor uses, here on blocks of runs at once; measured to agree with one run at a
    time within 1e-4 per hour); False the fast solver (within about 0.3 per hour)."""
    S = _system(system)
    X = np.atleast_2d(np.asarray(X, float))
    P, VA, VB, VI, T = unscale(X).T
    if exact:
        F, F0 = np.full((len(X), S.ns), np.nan), S.inlet_flows(VA, VB, VI)

        def solve(rows):
            try:
                F[rows] = S.simulate(P[rows], VA[rows], VB[rows], VI[rows], wcat, T[rows], truth=True,
                                     method="lsoda", a_spec=a_spec)[0]
            except RuntimeError:                    # one stubborn run must not lose the block
                if len(rows) > 1:
                    for r in rows:
                        solve(np.array([r]))
        for k in range(0, len(X), _BLOCK):
            solve(np.arange(k, min(k + _BLOCK, len(X))))
    else:
        F, F0 = S.simulate(P, VA, VB, VI, wcat, T, truth=True, a_spec=a_spec)
    with np.errstate(invalid="ignore"):
        XA, SC, _ = S.kpis(F, F0)
        return dict(profit=S.profit(F, F0, P, wcat), selectivity=SC, conversion=XA)


def _fingerprint(S):
    """Changes whenever the placeholder kinetics, prices or the box change, so that a stale
    cached curve is never used."""
    txt = repr((S.theta_true.round(10).tolist(), S.K_inhib, S.e_inhib, S.K_A, S.nu.tolist(), S.fwd.tolist(),
                S.thermo, sorted((k, repr(v)) for k, v in S.prices.items()), LOW.tolist(), HIGH.tolist(),
                MIN_TOTAL_FLOW, S.rev.tolist(), S.species, S.iC))
    return hashlib.md5(txt.encode()).hexdigest()[:10]


def true_tradeoff(system, n_points=12, wcat=WCAT, a_spec=A_SPEC, lo=None, hi=None, fixed=None, n_candidates=4096,
                  seed=0, min_profit=0.0, recompute=False, cache=True):
    """The TRUE profit versus selectivity curve of the virtual reactor (noise-free, fresh
    catalyst), for scoring only: the advisor never sees it.

    Same search as tradeoff(), with more candidates and finer refinement rounds.  The fast solver
    is only used to throw away settings that are far below the part of the curve that is shown
    (it is good to about 0.3 per hour; anything within 1 per hour of that part is kept); every
    number the curve is built from comes from the exact solver.  Returns the same kind of dict
    as tradeoff() (points with settings, profit, selectivity, conversion; markers; curve), plus
    curve_scoring: the curve at full density, continued a little below the profit floor, used
    by compare_with_truth.

    Takes about 20 to 40 s, so results are remembered: in memory for the session and, for the
    default box (whole ranges, no fixed knob), in smart/pareto_true.json like the true optima
    in smart/optima.json.  recompute=True ignores both; cache=False leaves the file alone."""
    S = _system(system)
    lo_, hi_, fixed_, free, pin = _box(lo, hi, fixed)
    default_box = lo is None and hi is None and not fixed_
    key = (f"{S.name}|W={float(wcat):g}|a={float(a_spec):g}|n={int(n_points)}|c={int(n_candidates)}|s={int(seed)}"
           f"|min={min_profit}|{_fingerprint(S)}")
    mkey = key + repr((lo_.tolist(), hi_.tolist(), sorted(fixed_.items())))
    if not recompute and mkey in _TRUE_MEMO:
        return _TRUE_MEMO[mkey]
    disk = {}
    if cache and default_box and os.path.exists(_TRUE_FILE):
        with open(_TRUE_FILE) as fh:
            disk = json.load(fh)
        if key in disk and not recompute:
            _TRUE_MEMO[mkey] = disk[key]
            return disk[key]
    t0 = time.time()
    rng = np.random.default_rng(seed)
    seen = dict(p_max=-np.inf, n=0, n_exact=0)

    def lowest(p_max):                              # the search refines down to here (see _search)
        fl = _floor(p_max, min_profit)
        return fl - 0.25 * (p_max - fl) if np.isfinite(fl) else -np.inf

    def evalf(C):
        tv = true_values(S, C, wcat, a_spec, exact=False)
        fast = np.where(np.isfinite(tv["profit"]), tv["profit"], -np.inf)
        seen["p_max"] = max(seen["p_max"], float(fast.max()))
        go = fast >= 2.0 * min(lowest(seen["p_max"]), 0.0) - 1.0      # generous: the top of the curve is not known yet
        out = {k: np.full(len(C), np.nan) for k in tv}
        if go.any():
            ex = true_values(S, C[go], wcat, a_spec, exact=True)
            for k in out:
                out[k][go] = ex[k]
        seen["n"] += len(C)
        seen["n_exact"] += int(go.sum())
        return out["profit"], out["selectivity"], out["conversion"], go

    C, p, s, xa = _search(evalf, pin, lo_, hi_, free, None, n_candidates, rng, min_profit, TRUE_ROUNDS,
                          n_anchor=TRUE_ANCHOR, n_local=TRUE_LOCAL)
    if not len(p):
        raise ValueError("no feasible setting in the box")
    F = _shown_front(p, s)
    scoring = F[p[F] >= lowest(p[F[0]])]
    scoring = scoring[_spread(p[scoring], s[scoring], 240)]
    Fk, sel_pos, knee_pos, fl = _pick(p, s, n_points, min_profit)
    points = [dict(**_settings(C[i]), x=[float(v) for v in C[i]], profit=_num(p[i], DEC_P),
                   selectivity=_num(s[i], DEC_S), conversion=_num(xa[i], DEC_S)) for i in Fk[sel_pos]]
    markers = _markers(points, knee_pos, sel_pos)
    out = dict(ok=True, system=S.name, product=S.species[S.iC], truth=True, wcat=float(wcat), a_spec=float(a_spec),
               points=points, markers=markers, curve=_curve(Fk, p, s),
               curve_scoring=dict(selectivity=[_num(v, DEC_S) for v in s[scoring]],
                                  profit=[_num(v, DEC_P) for v in p[scoring]]),
               f_best=_num(p[Fk[0]], DEC_P), n_evaluated=int(seen["n"]), n_exact=int(seen["n_exact"]),
               seconds=_num(time.time() - t0, 2),
               settings=dict(n_points=int(n_points), n_candidates=int(n_candidates), seed=int(seed),
                             min_profit=None if min_profit is None else float(min_profit)))
    _TRUE_MEMO[mkey] = out
    if cache and default_box:
        stem = key.rsplit("|", 1)[0] + "|"          # same system and settings, older kinetics / prices / box: stale
        disk = {k: v for k, v in disk.items() if not k.startswith(stem)}
        disk[key] = out
        tmp = _TRUE_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(disk, fh, indent=1)
        os.replace(tmp, _TRUE_FILE)
    return out


def compare_with_truth(result, system=None, wcat=None, a_spec=None, truth=None, top=0.8):
    """How close a predicted trade-off curve is to the true one.  Scoring only.
    wcat, a_spec: the catalyst load the truth is computed for; default = the one the curve was
    predicted for (recorded in the result), so the two can never silently differ.  The true
    curve is the one of the WHOLE box: for an advisor with narrower ranges or fixed knobs pass
    truth=true_tradeoff(system, lo=..., hi=..., fixed=...).

    For every predicted point the virtual reactor is run noise-free at those settings, giving
      profit_true, selectivity_true, conversion_true
      profit_error       predicted - true profit at the same settings
      selectivity_error  predicted - true selectivity at the same settings
      shortfall          how far the settings fall short of the true curve: the best true profit
                         available at that TRUE selectivity or better, minus the true profit
                         (0 = the point really is on the true curve); also in % of the true best
      curve_gap          predicted profit minus the true curve's profit at the same PREDICTED
                         selectivity: the vertical distance between the two drawn curves
    (None where the selectivity lies beyond the end of the true curve.)
    Summaries (mean and largest absolute values) are given for all points and for the
    profitable end: the points predicted to earn at least `top` of the predicted maximum."""
    S = _system(system if system is not None else result["system"])
    wcat = float(result.get("wcat", WCAT) if wcat is None else wcat)
    a_spec = float(result.get("a_spec", A_SPEC) if a_spec is None else a_spec)
    truth = truth if truth is not None else true_tradeoff(S, wcat=wcat, a_spec=a_spec)
    cv = truth.get("curve_scoring", truth["curve"])
    cs, cp = np.array(cv["selectivity"], float), np.array(cv["profit"], float)
    pts = result["points"]
    tv = true_values(S, np.array([pt["x"] for pt in pts]), wcat, a_spec, exact=True)
    pp, ps = np.array([pt["profit"] for pt in pts], float), np.array([pt["selectivity"] for pt in pts], float)
    f_best = float(cp[0])
    on_curve = lambda sel: np.interp(sel, cs, cp, left=cp[0], right=np.nan)
    short = on_curve(tv["selectivity"]) - tv["profit"]
    gap = pp - on_curve(ps)
    rows = dict(profit_true=tv["profit"], selectivity_true=tv["selectivity"], conversion_true=tv["conversion"],
                profit_error=pp - tv["profit"], selectivity_error=ps - tv["selectivity"],
                shortfall=short, shortfall_pct=100 * short / f_best, curve_gap=gap)

    def summary(m):
        out = dict(n=int(m.sum()), n_beyond_true_curve=int(np.isnan(short[m]).sum()))
        for k in ("profit_error", "selectivity_error", "shortfall", "shortfall_pct", "curve_gap"):
            v = np.abs(rows[k][m])
            v = v[np.isfinite(v)]
            out[f"{k}_mean_abs"] = _num(v.mean()) if len(v) else None
            out[f"{k}_max_abs"] = _num(v.max()) if len(v) else None
        return out

    i0 = result["markers"]["most_profitable"]
    return dict(
        f_best_true=_num(f_best), top=float(top),
        points=[{k: _num(v[i], 5) for k, v in rows.items()} for i in range(len(pts))],
        all=summary(np.ones(len(pts), bool)),
        profitable_end=summary(pp >= pp.max() - (1.0 - top) * abs(pp.max())),      # also right when the best is a loss
        best_point=dict(profit_predicted=_num(pp[i0]), profit_true=_num(tv["profit"][i0]),
                        pct_of_true_best=_num(100 * tv["profit"][i0] / f_best, 2),
                        selectivity_predicted=_num(ps[i0]), selectivity_true=_num(tv["selectivity"][i0])))
