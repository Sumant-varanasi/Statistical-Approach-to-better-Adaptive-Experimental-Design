"""
The smart (physics-informed) reactor advisor, and the baselines it is compared with.

Plain Bayesian optimisation (bo/) treats the reactor as a black box: it only sees one number
per experiment (the profit) and learns the landscape from scratch.  The advisor instead
knows the PHYSICS of the reactor and learns only what it does not know:

  KNOWN (built in)                         LEARNED from the experiments
  ---------------------------------------  ---------------------------------------------
  which reactions happen, stoichiometry    rate constants k_ref of every reaction
  mole balances over the catalyst surface  activation energies Ea (temperature response)
  Arrhenius temperature dependence         product adsorption (inhibition) and its heat
  ideal-gas partial pressures              catalyst activity loss with time-on-stream
  equilibrium limit from thermodynamics    a correction for anything the physics misses
  the economics (prices)                   (Gaussian process on the profit residual)

Every experiment gives the full GC composition (a mole fraction per species), so one
experiment teaches the model about EVERY reaction at once, not just "profit was 12.3".

How a campaign goes (same for every method, because that is how the reactor is run):
  * the catalyst is loaded once (WCAT) and never changed
  * the operator's 5 screening runs come first (systems.screening_runs)
  * from then on every suggested run must lie within MAX_STEP of the previous run for every
    knob, inside the safety box, and leave enough total flow for the GC
  * optionally a check run (the base case repeated) every few runs, to see drift

How the advisor chooses the next run (v5, planner="horizon"):
  1. fit the kinetics to all GC readings so far (kinetics.py), with its uncertainty, choosing
     the model class (inhibition, its temperature dependence, deactivation) the data support
  2. draw a handful of plausible versions of the chemistry consistent with the data
  3. PLAN: find where the model expects the best profit over the whole box (x*)
  4. for the candidate first steps within reach, roll the model forward H runs (greedy on the
     model mean, one allowed step at a time) and score the whole trajectory under every
     plausible chemistry:  return = sum of the H profits + (R - H) x the best profit found,
     R = runs the operator can still afford.  So a step that earns nothing now but opens a
     better region is chosen when the campaign is long enough to cash it in, and a safe
     step is chosen when it is not.
  5. a Gaussian-process correction, fitted to what the physics got wrong so far, is added to
     every prediction (with a noise floor from the GC error), and the predictive spread is
     calibrated against the advisor's own past misses
  6. product specs (minimum selectivity, maximum adiabatic temperature rise, ...) are
     enforced by the probability, over the plausible chemistries, that a run meets them
planner="greedy" keeps the v4 rule (Expected Improvement one step ahead, with a purposeful
step toward x* when nothing within reach beats the best run), for comparison.

v7 adds two things, both off by default (so v5 and v6 are reproduced unchanged):
  * correction="hybrid": the v6 committee plus a stand-alone neural network that predicts the
    profit from the knobs alone (smart/committee.py)
  * revise=True: the advisor no longer takes its reaction network for granted.  After every
    run it checks whether the network still explains the GC readings; when it does not, it
    asks smart/discovery.py for the best SINGLE change (add / remove / replace a reaction, or
    change one reaction order) and adopts it only on strong evidence.  Every change is written
    down in plain words (SmartAdvisor.revisions).  See SmartAdvisor._revise.
"""
import hashlib
import json
import os
import time
import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution, minimize
from .systems import (SYSTEMS, MODELS, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, WCAT, A_SPEC, DIM, R, T_REF,
                      unscale, scale, physical, feasible, step_scaled, base_case, screening_runs)
from .kinetics import KineticFit, GC_NOISE_REL, GC_NOISE_ABS, HOURS_PER_RUN, PRIOR_E, GC_REL, GC_ABS
from .committee import ResidualCommittee, ProfitEnsemble, MEMBERS, HYBRID_MEMBERS, ALL_MEMBERS
from . import discovery as D
from bo.gp import GP
from bo.loop import LogInputs, copula_transform
from bo.acquisition import expected_improvement


# =================================================================================== the virtual reactor
class VirtualReactor:
    """Stands in for the real reactor + GC.  Uses the TRUE kinetics (with the adsorption terms
    the advisor's model does not fully have) and adds GC noise:
        y_GC = y_true * (1 + 1 % noise) + 0.05 % absolute noise, renormalised
    (typical repeatability of an online GC with an internal standard).
    The operator never sees profit directly: it is computed from the GC reading using the
    inert as an internal standard, exactly as the advisor tool does with real data.

    deactivation: base rate of activity loss per hour on stream (0 = a stable catalyst).  The
    true loss is severity-weighted (faster when hotter and with more A in the feed, as coking
    is), a form the advisor's model does not know: it assumes a constant rate."""

    def __init__(self, system, seed=0, wcat=WCAT, a_spec=A_SPEC, gc_rel=GC_NOISE_REL, gc_abs=GC_NOISE_ABS,
                 deactivation=0.0, hours_per_run=HOURS_PER_RUN):
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.rng = np.random.default_rng(50_000 + seed)
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.gc_rel, self.gc_abs = gc_rel, gc_abs
        self.deactivation, self.hours_per_run = float(deactivation), float(hours_per_run)
        self.ln_activity, self.t_on_stream, self.n_runs = 0.0, 0.0, 0

    def activity(self):
        return float(np.exp(self.ln_activity))

    def loss_rate(self, x):
        """True activity loss per hour at settings x: Arrhenius in T (E = 40 kJ/mol) and
        growing with the partial pressure of A (coke precursor), 1x at 10 bar with 40 % A."""
        P, VA, VB, VI, T = unscale(x)[0]
        PA = P * VA / (VA + VB + VI)
        return self.deactivation * np.exp(-40e3 / R * (1 / T - 1 / T_REF)) * np.sqrt(max(PA, 0.0) / 4.0)

    def true_flows(self, x, ln_act=None):
        P, VA, VB, VI, T = unscale(x)[0]
        act = np.exp(self.ln_activity if ln_act is None else ln_act)
        F, F0 = self.S.simulate(P, VA, VB, VI, self.wcat, T, truth=True, method="lsoda", a_spec=self.a_spec * act)
        return F, F0, P

    def true_profit(self, x, ln_act=None):
        """True profit of settings x at the current activity (or at ln_act)."""
        F, F0, P = self.true_flows(x, ln_act)
        return float(self.S.profit(F, F0, [P], [self.wcat])[0])

    def run(self, x, hours=None):
        hours = self.hours_per_run if hours is None else float(hours)
        F, F0, P = self.true_flows(x)
        y = F / F.sum(1, keepdims=True)
        y = y * (1 + self.gc_rel * self.rng.standard_normal(y.shape)) \
            + self.gc_abs * self.rng.standard_normal(y.shape)
        y = np.maximum(y, 0.0)
        y /= y.sum(1, keepdims=True)
        ph6 = physical(x, self.wcat)
        XA, SC, _ = kpis_from_gc(self.S, ph6, y)
        out = dict(gc=y[0], profit=float(profit_from_gc(self.S, ph6, y)[0]),
                   profit_true=float(self.S.profit(F, F0, [P], [self.wcat])[0]),
                   X_A=float(XA[0]), S_C=float(SC[0]), hours=hours, activity=self.activity(),
                   dT_ad=float(self.S.dT_ad(F, F0)[0]))
        self.ln_activity -= self.loss_rate(x) * hours     # the catalyst ages during the run
        self.t_on_stream += hours
        self.n_runs += 1
        return out


def profit_from_gc(S, phys6, y):
    """Profit from settings + a GC reading (inert = internal standard).  This is all the
    operator has to provide: what they set, and what the GC printed.
    phys6 rows: P, V_A, V_B, V_I, T, Wcat."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    F = S.flows_from_gc(y, VA, VB, VI)
    F0 = S.inlet_flows(VA, VB, VI)
    return S.profit(F, F0, P, W)


def kpis_from_gc(S, phys6, y):
    """Conversion, selectivity and yield as the operator would read them off the GC."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    return S.kpis(S.flows_from_gc(y, VA, VB, VI), S.inlet_flows(VA, VB, VI))


def dT_ad_from_gc(S, phys6, y):
    """Adiabatic temperature rise implied by a GC reading (what the isothermal model cannot see)."""
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    P, VA, VB, VI, T, W = phys6.T
    return S.dT_ad(S.flows_from_gc(y, VA, VB, VI), S.inlet_flows(VA, VB, VI))


def profit_noise_sd(S, phys6, y, n=64, seed=0):
    """How much GC noise alone moves the profit computed from a reading (Monte Carlo with
    the GC error model).  Profit is a small difference of large numbers, so 1 % GC noise
    becomes several money units per hour; the correction GP must not mistake that for a
    real effect."""
    rng = np.random.default_rng(seed)
    phys6, y = np.atleast_2d(phys6), np.atleast_2d(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        Yn = y[i] * (1 + GC_NOISE_REL * rng.standard_normal((n, y.shape[1]))) \
            + GC_NOISE_ABS * rng.standard_normal((n, y.shape[1]))
        Yn = np.maximum(Yn, 0)
        Yn /= Yn.sum(1, keepdims=True)
        out[i] = profit_from_gc(S, np.repeat(phys6[i:i + 1], n, 0), Yn).std()
    return out


class SmoothGP(GP):
    """GP for the physics correction: length-scales of at least 0.15 (in scaled units).
    What the physics gets wrong (here: product inhibition) changes slowly across the box;
    anything wigglier than that is measurement noise, not chemistry."""
    LS_BOUNDS = (np.log(0.15), np.log(20.0))


# =================================================================================== step limits
def step_box(x_prev, step, lo, hi, fixed=None):
    """Scaled box of runs reachable from x_prev in one step, inside the safety box."""
    blo = np.maximum(lo, x_prev - step)
    bhi = np.minimum(hi, x_prev + step)
    for i, v in (fixed or {}).items():
        blo[i] = bhi[i] = v
    return blo, np.maximum(bhi, blo)


def sobol_in(lo, hi, n, rng, free=None):
    """n Sobol points in the box; with `free` (indices), only those dimensions vary."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if free is None or len(free) == DIM:
        return lo + qmc.Sobol(DIM, scramble=True, seed=int(rng.integers(1 << 31))).random(n) * (hi - lo)
    out = np.tile(lo, (n, 1))
    if len(free):
        out[:, free] = lo[free] + qmc.Sobol(len(free), scramble=True, seed=int(rng.integers(1 << 31))).random(n) * (hi - lo)[free]
    return out


def keep_feasible(C):
    return C[feasible(unscale(C))]


# =================================================================================== the advisor
COMMITTEES = {"committee": MEMBERS, "hybrid": HYBRID_MEMBERS}     # correction -> its default committee members
# saved with every campaign of the hybrid committee or of network revision (run_smart); benchmark.is_stale
# treats a cached file with another value as made by an earlier definition.  Change it whenever v7 changes.
V7_VERSION = "v7 of 2026-11-25 (GP refitted per fold next to the stand-alone network; revision screened with site " \
             "blocking, held on one bad run, never repeated on the same runs)"
SPEC_KEYS = {"S_C": ("min", "selectivity"), "X_A": ("min", "conversion"), "dT_ad": ("max", "adiabatic temperature rise")}


def _what_was_wrong(m):
    """A failed network check (discovery.misfit) in plain words, for the record of a revision."""
    if m["not_made"]:
        s = m["not_made"]
        return (f"the model could not explain {' and '.join(s)}: none of its reactions makes {'it' if len(s) == 1 else 'them'}, "
                "yet the GC sees up to " + " and ".join(f"{m['detected'][x]:.1f} mol-% {x}" for x in s))
    if m["beyond"]:
        return ("the model missed " + ", ".join(f"{x} by {m['miss'][x]['x_noise']:.1f}" for x in m["beyond"])
                + " times the GC noise, more than the allowance for model error")
    if not m["within_allowance"]:
        return ("the model's misses were larger than the GC noise plus the allowance for model error "
                f"({m['x_noise']:.1f} times the GC noise overall)")
    return ("the model missed " + ", ".join(f"{x} by {m['miss'][x]['x_noise']:.1f}" for x in m["unexplained"])
            + " times the GC noise, within the allowance for model error but more than twice the GC noise")


class SmartAdvisor:
    """
    adv = SmartAdvisor('complex', wcat=1.0, a_spec=100)
    adv.update(X, gc, profit, hours=...)   X scaled (n x 5), gc (n x ns), measured profit (n,)
    x, info = adv.suggest(x_prev)          next run (scaled), reachable from x_prev, and why
    adv.predict(Xc)                        predicted profit mean and sd at candidates
    adv.recommend()                        index of the best tested run (by prediction, meeting the spec)
    adv.plan()                             where the model expects the best profit (x*, mean, sd)

    wcat, a_spec  catalyst mass (g) and BET area (m2/g) of the loaded bed
    max_step      largest change per knob between consecutive runs (physical units)
    lo, hi        optional safety box in scaled units (e.g. never above 700 K)
    fixed         {knob index: scaled value} knobs the operator cannot change
    spec          product / safety spec, e.g. {"S_C": 0.9, "dT_ad": 60}: minimum selectivity,
                  maximum adiabatic temperature rise (K); also "X_A" (minimum conversion)
    planner       "horizon" (v5 lookahead) or "greedy" (v4 one-step rule)
    horizon       runs the lookahead rolls forward (H)
    remaining     runs the operator can still afford (R); the planner weighs exploration by it
    residual_gp   correct the physics on the profit residual (on by default; False = physics only)
    correction    "gp" (v5: one Gaussian process), "committee" (v6: none / GP / neural-network
                  ensemble, weighted by how well each predicts held-out runs; smart/committee.py)
                  or "hybrid" (v7: the v6 committee plus a stand-alone neural network that
                  predicts the profit from the knobs alone, without the physics; the advisor
                  trusts whichever has predicted unseen runs better)
    members       committee members to use, a subset of committee.ALL_MEMBERS containing "none";
                  None = the default of the chosen correction (v6: none, gp, nn; v7: all four).
                  Leaving one out shows what it contributes.
    calibrate     scale the predictive spread by the advisor's own past misses
    models        kinetic model classes the fit may choose from (kinetics.MODELS); None = all
    revise        False (default): the reaction network given in `system` is never changed.
                  True (v7): after every update the advisor checks whether its network explains
                  the GC readings and, when it does not, may change it one step at a time (see
                  _revise).  adv.S is then the network in use, adv.S_start the one it was given,
                  adv.revisions the changes made (plain words), adv.misfit the last check (with
                  'bad_run' when the misfit is down to one reading to be checked), adv.revision_log
                  every search or decision (outcome 'revised', 'kept', 'held' for a reading to check,
                  'time limit', 'taken back' after an edited log), adv.revisions_undone the changes
                  taken back because the runs they rested on were edited.
    library       reactions the revision may use: None = discovery.reaction_library() (every
                  atom-balanced reaction among the species), or a list of reaction texts (read
                  when the advisor is built, so a typing error shows at once)
    revise_seconds  with revise=True: about how long the check-and-repair of the network may take
                  in one update (None = no limit, as in the benchmark).  It is checked between the
                  steps of a search and also limits the fresh search; one screening batch (5 to 30 s)
                  can run past it.  When it is hit the advisor keeps the network it has, says so, and
                  searches again at the next update.  For a web page that waits for the answer.
    """
    # ---- network revision (only used with revise=True)
    REVISE_MIN_RUNS = 5        # no revision before the 5 screening runs are in
    REVISE_GAIN = 10.0         # BIC must improve by this much to adopt a change (Kass & Raftery: > 10 is very strong evidence)
    REVISE_RETURN = 20.0       # ... and by this much to go BACK to a network the advisor has already left (hysteresis)
    REVISE_WAIT = 3            # after a search that changed nothing: runs to wait before searching again
    REVISE_STEPS = 3           # at most this many changes, one after the other, in one update
    REVISE_MAX_SEARCHES = 12   # at most this many searches by one advisor (keeps a campaign fast)
    REVISE_TOP = 3             # how many of the best single changes are compared like for like
    REVISE_FRESH_RUNS = 8      # a search from no reactions at all (the way out of a dead end) needs at least this many runs
    REVISE_FRESH_MAX = 2       # ... and is made at most this many times by one advisor
    REVISE_FRESH_SECONDS = 60  # ... each limited to about this long
    MAX_REACTIONS = 4          # the revision never builds a network with more reactions than this

    def __init__(self, system, wcat=WCAT, a_spec=A_SPEC, seed=0, n_theta=6, lo=None, hi=None,
                 fixed=None, max_step=None, residual_gp=True, spec=None, planner="horizon", horizon=4,
                 remaining=10, calibrate=True, models=None, correction="gp", members=None, revise=False,
                 library=None, revise_seconds=None):
        if correction not in (None, "gp") + tuple(COMMITTEES):
            raise ValueError(f"correction must be 'gp', 'committee' or 'hybrid', got {correction!r}")
        if members is not None and correction not in COMMITTEES:
            raise ValueError(f"members only applies to correction='committee' or 'hybrid' (got correction={correction!r}); "
                             "it would be ignored")
        self.S = SYSTEMS[system] if isinstance(system, str) else system
        self.wcat, self.a_spec = float(wcat), float(a_spec)
        self.kin = KineticFit(self.S, a_spec=self.a_spec, seed=seed, **({} if models is None else dict(models=models)))
        self.n_theta = n_theta
        self.rng = np.random.default_rng(seed)
        self.lo = np.zeros(DIM) if lo is None else np.asarray(lo, float)
        self.hi = np.ones(DIM) if hi is None else np.asarray(hi, float)
        self.fixed = dict(fixed or {})
        self.free = [i for i in range(DIM) if i not in self.fixed]
        self.step = step_scaled(max_step)
        self.use_gp = residual_gp
        self.correction = correction if residual_gp else None
        self.members = None                     # committee members (v6 / v7 only)
        if self.correction in COMMITTEES:
            self.members = ResidualCommittee(members=COMMITTEES[correction] if members is None else members).members
        self.committee = None
        self.spec = {k: float(v) for k, v in (spec or {}).items() if v is not None and k in SPEC_KEYS}
        self.planner, self.horizon, self.remaining = planner, int(horizon), int(remaining)
        self.calibrate = calibrate
        self.calib = 1.0
        self.gp = None
        self.seed = seed
        self.n = 0
        self.t_next = 0.0
        # ---- v7: revision of the reaction network (nothing below is touched unless revise=True)
        self.revise, self.library = bool(revise), library
        if self.revise and [s for s in self.S.species if s not in D.SPECIES_TABLE]:
            raise ValueError("revise=True needs the species of smart/discovery.py (A, B, C, D, E and the inert I); this "
                             f"system has {', '.join(self.S.species)}")
        self.S_start = self.S                   # the network the advisor was given; also supplies species, prices, product
        self.revisions = []                     # every change of the network, in plain words (see _revise_once)
        self.revision_log = []                  # every search, also those that changed nothing
        self.misfit = None                      # the last check: how well the network explains the GC (discovery.misfit)
        self.n_searches, self.n_fresh, self._next_search, self._left, self._lib = 0, 0, 0, {}, None
        self.revise_seconds = None if revise_seconds is None else float(revise_seconds)
        self.revisions_undone = []              # changes taken back because the runs they were based on were edited
        self.progress = None                    # optional callback during a search (update(progress=...))
        self._searched_on, self._basis, self._t_rev, self._timed_out = None, None, None, False
        if self.revise:
            self._library()                     # read the library now: a typing error in it shows when the advisor is built

    # ------------------------------------------------------------------ learning
    def update(self, X, gc, profit, wcat_rows=None, hours=None, progress=None):
        """wcat_rows: catalyst mass of each logged run (defaults to the campaign's wcat);
        hours: duration of each run (time-on-stream clock; default 24 h each).
        With revise=True the reaction network may change here (see _revise); the kinetics, the
        plausible chemistries and the correction below are then those of the new network.
        progress: optional callback, called during a search for another network with a dict
        {stage, message, elapsed_s} (revise=True only; a search can take a minute)."""
        self.X = np.atleast_2d(np.asarray(X, float))
        self.gc = np.atleast_2d(gc)
        self.y = np.asarray(profit, float)
        n = len(self.y)
        self.n = n
        self.W = np.broadcast_to(np.asarray(self.wcat if wcat_rows is None else wcat_rows, float), (n,)).copy()
        self.hours = np.full(n, HOURS_PER_RUN) if hours is None else np.asarray(hours, float)
        ph6 = physical(self.X, self.W)
        if self.revise:
            self.progress = progress
            self._replay_if_edited()
        n_starts = None if self.kin.n_fits == 0 else (3 if n % 5 == 0 else 1)
        self.kin.fit(ph6, self.gc, hours=self.hours, n_starts=n_starts)
        if self.revise:
            self._revise(ph6)
        self.t_rows = self.kin.t_rows
        self.t_next = float(self.t_rows[-1] + self.hours[-1])
        self.thetas = self.kin.samples(self.n_theta, self.rng)
        self.phys_at_data = self._physics(self.X, self.kin.theta[None], W=self.W, t=self.t_rows)[0]
        self.resid = self.y - self.phys_at_data
        self.gp = None
        self.committee = None
        if self.correction and n >= 4:
            # the GP's noise is not allowed to drop below what the GC noise implies
            s = max(self.resid.std(), 1e-6)
            noise_var = float(np.mean(profit_noise_sd(self.S, ph6, self.gc) ** 2))
            nf = noise_var / s ** 2
            nb = (min(nf, 0.5), max(1.0, 2 * nf))
            if self.correction in COMMITTEES:
                self.committee = ResidualCommittee(seed=self.seed, members=self.members).fit(
                    self.X, self.resid, self.phys_at_data, noise_var, gp_noise_bounds=nb)
            else:
                gp = SmoothGP(seed=self.seed, noise_bounds=nb)
                self.gp = LogInputs(gp, (0,), LOW, HIGH).fit(self.X, self.resid)     # GP sees log P
        # calibration: z-scores of the measured profits against the predictive spread of an
        # observation at the data (physics disagreement + GP, noise included); inflate if > 1
        self.calib = 1.0
        if self.calibrate and n >= 6:
            mu, sd, _ = self._predict(self.X, t=self.t_rows, include_noise=True)
            z = (self.y - mu) / np.maximum(sd, 1e-9)
            self.calib = float(np.clip(np.sqrt(np.mean(z ** 2)), 1.0, 5.0))
        self.mu_data = self.predict(self.X, t=self.t_rows)[0]
        return self

    # ------------------------------------------------------------------ the reaction network (v7: revise=True)
    def network(self):
        """The reactions the advisor is working with, one text line each ('A + B <=> C', ...;
        the format of smart/discovery.py, so it can be typed back in)."""
        return D.network_text(self.S)

    def explains(self, ph6=None):
        """How well the advisor's reaction network, as fitted, explains the GC readings logged so
        far (discovery.misfit): the typical miss per species as a multiple of the GC noise
        (x_noise), the species no reaction makes (not_made), the species missed by more than the
        allowance for model error (beyond) and `within_allowance`: True = the network is taken as
        adequate.  'words' says it in one plain sentence.  It reads the kinetic fit and changes
        nothing, so it can be asked of any advisor (v5, v6 or v7) after update()."""
        ph6 = physical(self.X, self.W) if ph6 is None else ph6
        yp = self.kin.predict_gc(ph6, t=self.kin.t_rows)
        m = D.misfit(self.S, D.make_data(ph6, self.gc, self.hours, self.a_spec), yp, len(self.kin.theta))
        m["bad_run"] = None                     # filled in by _revise when the misfit is down to one run (_bad_run)
        m["words"] = ("the reaction network explains the GC readings within the GC noise "
                      f"(typical miss {m['x_noise']:.1f} times the noise)" if m["fits"] else
                      "the reaction network explains the GC readings within the GC noise plus the allowance for what "
                      f"any textbook model leaves out (typical miss {m['x_noise']:.1f} times the GC noise)"
                      if m["within_allowance"] else f"the reaction network does not explain the GC readings: {_what_was_wrong(m)}")
        return m

    def _ladder(self, S, ph6, warm):
        """The kinetic fit of network S to the runs so far, with the same model classes the
        advisor itself uses.  Two fits: one started from fitted plain kinetics `warm` (+ one
        random start), one from the chemist's starting guess of S; the one with the better
        evidence (_evidence) is kept.  The second guards against constants that were fitted to a
        wrong network and are bent to make up for what it lacks.  None if both failed."""
        best = None
        for w in (warm, None):
            kf = KineticFit(S, a_spec=self.a_spec, seed=self.seed, models=self.kin.models)
            if w is not None:
                kf._warm = {"plain": np.asarray(w, float)[:2 * S.nr]}
            kf.fit(ph6, self.gc, hours=self.hours, n_starts=2 if w is not None else 1)
            if kf.n_fits and (best is None or self._evidence(kf) < self._evidence(best)):
                best = kf
        return best

    def _evidence(self, kf):
        """BIC of a kinetic fit of the runs so far, counted with the fit's own weights (GC noise
        plus the allowance for model error): chi2 + (number of fitted parameters) x ln(number of
        data).  Lower is better.  The same measure for every network, so differences compare
        like with like; the allowance keeps a small imperfection of a right network (an
        adsorption term, a little ageing) from counting as evidence for another reaction."""
        n_data, p = self.n * (self.S_start.ns - 1), len(kf.theta)
        return float(kf.chi2_dof * max(n_data - p, 1) + (p + ("d" in kf.extras())) * np.log(n_data))

    def _revise(self, ph6):
        """v7: does the reaction network still explain what the GC prints?  If not, repair it.

        The check (after every update, cheap): the misses of the advisor's own kinetic fit
        (explains).  The network is left alone while every product the GC sees is made by some
        reaction, the misses stay within the GC noise plus the allowance for model error that
        the kinetic fit itself uses (2 % of the reading + 0.2 mol-%), and no species is missed by
        more than twice the GC noise (what discovery calls 'cannot explain').  A right network on
        an imperfect model (an adsorption term it lacks, a catalyst that ages) normally passes.
        A reaction the network lacks, or a misfit larger than the allowance, does not; but a
        superfluous or slightly wrong reaction can pass (there is no step that removes a reaction
        while the network passes the check: see the open issues).
        One bad reading is not a reason to change the chemistry: when the misfit sits in ONE run
        and the other runs are explained within the allowance (a typing error, a bad GC injection
        look like that), the advisor asks for that reading to be checked and does not search
        (explains() says which run and which reading).
        When the network fails the check, and a search is due, discovery.one_step_revision
        tries every single change of it (add / remove / replace a reaction from the enumerated
        library, or change one reaction order) on the campaign's own runs, with plain rate laws
        and, when the advisor's own fit found product C blocking catalyst sites, also with site
        blocking at that strength (so that a change that only imitates the site blocking cannot
        push the right one out).  The best few are
        then compared with the network as it stands LIKE FOR LIKE: the same runs, the same
        kinetic model classes (kinetics.KineticFit), the same measure (_evidence: BIC counted
        with the weights of the kinetic fit), each fitted both from the constants fitted so far and
        from the starting guess (constants fitted to a wrong network are bent to make up for what
        it lacks, and from them the right network can stop at a poor fit: seen in a replay of a
        logged campaign, 8 times the GC noise against 0.6).  A change is adopted only when it
        lowers that BIC by REVISE_GAIN (10: very strong evidence) or more.  Then the kinetics of the new network
        replace the old ones, and update() goes on to refit the plausible chemistries and the
        correction with them.  Up to REVISE_STEPS changes are made in one update, each on the
        same evidence rule, so a badly wrong network is repaired in a few steps.

        When no single change is clearly better although the network still fails the check, the
        advisor may be in a dead end; from REVISE_FRESH_RUNS runs on it then searches the reactions
        afresh from none at all, at most REVISE_FRESH_MAX times (_start_again).

        So that the advisor does not flip back and forth: going back to a network it has already
        left needs twice the evidence (REVISE_RETURN).  So that a campaign stays fast: after a
        search that changed nothing the advisor waits REVISE_WAIT runs before it searches again,
        and it never searches more than REVISE_MAX_SEARCHES times.  The same runs are never
        searched twice: asking again (update with the same log) gives the same network.  When
        rows of the log that a change was based on are edited or removed, the changes are taken
        back and the check starts again from the network the advisor was given
        (_replay_if_edited), so the network follows from the runs as they are now.
        revise_seconds (optional) limits the time spent here in one update.
        The profit and the KPIs of the runs are computed from the measured GC readings and do not
        depend on the network (the adiabatic rise read off the GC uses the network the advisor
        was given); only the advisor's predictions do."""
        if not self.kin.scores or self.n < self.REVISE_MIN_RUNS:
            return
        self._t_rev, self._timed_out = time.time(), False
        self.misfit = m = self.explains(ph6)
        fp = self._fingerprint()
        if fp == self._searched_on:                 # these very runs have been searched already: same answer
            return
        bad = self._bad_run(ph6, m) if self._needs_search(m) else None
        if bad is not None:
            m["bad_run"] = bad
            m["words"] = (f"the reaction network explains every run except run {bad['run']} within the GC noise plus the "
                          f"allowance for what any textbook model leaves out; in run {bad['run']} the {bad['species']} reading "
                          f"({bad['measured_mol_pct']:.2f} mol-%) is far from what the reactions fitted to the other runs predict "
                          f"({bad['predicted_mol_pct']:.2f} mol-%): check run {bad['run']}'s GC printout (a typing error, a bad "
                          "injection) before trusting it")
            self.revision_log.append(dict(run=int(self.n), network=self.network(), miss_x_noise=m["x_noise"],
                                          verdict=m["verdict"], outcome="held", seconds=time.time() - self._t_rev, words=(
                f"After run {self.n}: {m['words']}. The advisor does not change its reactions because of one reading; if the "
                "reading is right, the next runs near those settings will show the same and it will search then.")))
            return
        steps, searched = 0, False
        while (self._needs_search(m) and self.n >= self._next_search and steps < self.REVISE_STEPS
               and self.n_searches < self.REVISE_MAX_SEARCHES and not self._out_of_time()):
            searched = True
            if not self._revise_once(ph6, m):
                if self._timed_out:
                    break
                # no single change helps: a dead end, or simply not enough evidence yet (see _start_again)
                if self.misfit["within_allowance"] or not self._start_again(ph6, self.misfit):
                    if not self._timed_out:
                        self._next_search = self.n + self.REVISE_WAIT
                    break
            steps += 1
            self.misfit = m = self.explains(ph6)
        if self._timed_out or (self._out_of_time() and self._needs_search(m) and self.n >= self._next_search
                               and steps < self.REVISE_STEPS and self.n_searches < self.REVISE_MAX_SEARCHES):
            self._timed_out = True
            self.revision_log.append(dict(run=int(self.n), network=self.network(), miss_x_noise=m["x_noise"],
                                          verdict=m["verdict"], outcome="time limit", seconds=time.time() - self._t_rev, words=(
                f"After run {self.n}: the search for a better reaction network was stopped by the time limit "
                f"({self.revise_seconds:g} s); the advice uses the network as it stands and the search goes on at the "
                "next update.")))
        elif searched:
            self._searched_on = fp

    REVISE_ON_UNEXPLAINED = True   # also search when a species is missed by more than twice the GC noise (within the allowance)

    def _needs_search(self, m):
        """Does the network fail the check (see _revise)?"""
        return bool(not m["within_allowance"] or (self.REVISE_ON_UNEXPLAINED and m["unexplained"]))

    def _bad_run(self, ph6, m):
        """Is a failed check down to ONE run, as a typing error or a bad GC injection would be?
        Asked only when every product the GC sees is made and one run carries most of the misfit
        (discovery.misfit: one_run).  The network is then fitted again WITHOUT that run (one kinetic
        fit; the run's hours stay on the clock so the later runs keep their time on stream).  It is
        a bad run when the other runs are then explained within the allowance and that run is not.
        A wrong or missing reaction does not pass this: the other runs still show it (checked on the
        screening runs of three campaigns: the two wrong candidate networks tried never passed, a
        reading set to zero always did).  Returns {run (from 1), species, measured_mol_pct,
        predicted_mol_pct, x_allow} for the reading missed most, or None."""
        if m["not_made"] or not m.get("one_run"):
            return None
        k = int(m["worst_run"]) - 1
        keep = np.arange(self.n) != k
        hk = self.hours.copy()
        if k > 0:
            hk[k - 1] += hk[k]
        kf = KineticFit(self.S, a_spec=self.a_spec, seed=self.seed, models=self.kin.models)
        kf.fit(ph6[keep], self.gc[keep], hours=hk[keep])
        if not kf.n_fits:
            return None
        rest = D.misfit(self.S, D.make_data(ph6[keep], self.gc[keep], hk[keep], self.a_spec),
                        kf.predict_gc(ph6[keep], t=kf.t_rows), len(kf.theta))
        g = np.asarray(self.gc[k], float)
        yk = kf.predict_gc(ph6[k:k + 1], t=KineticFit.time_on_stream(self.hours)[k:k + 1])[0]
        za = ((yk - g) / (GC_REL * g + GC_ABS))[:-1]
        x_allow = float(np.sqrt(np.mean(za ** 2)))
        if not rest["within_allowance"] or x_allow <= 1.0:
            return None
        i = int(np.argmax(np.abs(za)))
        return dict(run=k + 1, species=self.S.species[i], measured_mol_pct=float(100 * g[i]),
                    predicted_mol_pct=float(100 * yk[i]), x_allow=x_allow)

    def _out_of_time(self):
        return self.revise_seconds is not None and self._t_rev is not None and time.time() - self._t_rev > self.revise_seconds

    def _say(self, message):
        """Tell the caller's progress callback (update(progress=...)) what the search is doing."""
        if self.progress:
            self.progress(dict(stage="network", message=message,
                               elapsed_s=0.0 if self._t_rev is None else time.time() - self._t_rev))

    def _fingerprint(self):
        """A short digest of the log (settings, GC readings, hours, catalyst mass)."""
        h = hashlib.sha1()
        for a in (self.X, self.gc, self.hours, self.W):
            h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
        return (int(self.n), h.hexdigest())

    def _replay_if_edited(self):
        """The network must follow from the runs as they are NOW.  If a row of the log that a change
        of the network was based on has been edited or removed since (a corrected GC reading, a
        deleted run), every change is taken back: the advisor returns to the network it was given,
        with a fresh kinetic fit, and the check (and, if needed, the search) starts again from
        there.  The changes taken back are kept in revisions_undone."""
        if self._basis is None:
            return
        Xb, gb, hb = self._basis
        m = len(Xb)
        if (len(self.X) >= m and np.array_equal(self.X[:m], Xb) and np.array_equal(np.asarray(self.gc[:m], float), gb)
                and np.array_equal(self.hours[:m], hb)):
            return
        many = len(self.revisions) > 1
        self.revision_log.append(dict(run=int(self.n), network=self.network(), outcome="taken back", seconds=0.0, words=(
            f"The log was edited: a run that the change{'s' if many else ''} of the reaction network "
            f"({'; '.join(r['change'] for r in self.revisions)}) {'were' if many else 'was'} based on has been changed or "
            f"removed. The advisor goes back to the network it was given ({'; '.join(D.network_text(self.S_start))}) and "
            "checks it against the runs as they are now.")))
        self.revisions_undone += self.revisions
        self.revisions = []
        self.S = self.S_start
        self.kin = KineticFit(self.S_start, a_spec=self.a_spec, seed=self.seed, models=self.kin.models)
        self._basis, self._left, self._next_search, self._searched_on, self.misfit = None, {}, 0, None, None

    def _site_blocking(self, ph6):
        """K_C (1/bar) at the temperature of every run from the advisor's own kinetic fit, or None
        when the data did not support product C blocking catalyst sites."""
        parts = self.kin.split(self.kin.theta)
        if "lnK" not in parts:
            return None
        e = parts["e"][0] if "e" in parts else PRIOR_E
        return np.exp(parts["lnK"][0] + e * (T_REF / ph6[:, 4] - 1.0))

    def _build(self, reactions):
        """A revised network as a reactor model.  Species, prices and product come from the network
        the advisor was given; so do the heats of reaction (and, for a two-way reaction, its
        equilibrium) of every reaction it shares with it, so that a change does not alter what is
        known about the reactions that stay.  New reactions take theirs from discovery.SPECIES_TABLE."""
        S_new = D.build_system(reactions, base=self.S_start, name=f"{self.S_start.name} (revised)")
        old = {r.entry_key(): (j, r) for j, r in enumerate(D.parse_network(self.S_start))}
        for j, r in enumerate(S_new.reactions):
            if r.entry_key() not in old:
                continue
            k, q = old[r.entry_key()]
            sign = 1.0 if r.direction_key() == q.direction_key() else -1.0    # the same reaction written the other way round
            S_new.dH[j] = sign * self.S_start.dH[k]
            if r.reversible and self.S_start.thermo[k] is not None:
                S_new.thermo[j] = tuple(sign * v for v in self.S_start.thermo[k])
        return S_new

    def _library(self):
        """The reactions the revision may use: the library, limited to this system's species."""
        if self._lib is None:
            sp = set(self.S_start.species)
            self._lib = [r for r in D._library_reactions(self.library) if set(r.reactants) | set(r.products) <= sp]
        return self._lib

    def _revise_once(self, ph6, m):
        """One search for a single change of the network; True if a change was adopted."""
        t0 = time.time()
        data = D.make_data(ph6, self.gc, self.hours, self.a_spec)
        net = D.parse_network(self.S)
        self._say(f"after run {self.n}: trying every single change of the reaction network")
        res = D.one_step_revision(net, data, library=self._library(), theta=self.kin.theta_kin,
                                  max_reactions=self.MAX_REACTIONS, inhib=self._site_blocking(ph6))
        self.n_searches += 1
        log = dict(run=int(self.n), network=D.network_text(net), miss_x_noise=m["x_noise"], verdict=m["verdict"],
                   n_tried=res["n_tried"])
        # ---- the network as it stands, fitted once more from the search's own optimum: a change must
        # beat the BEST fit of the present network, not a fit that happens to be stuck
        bic0 = self._evidence(self.kin)
        self._say(f"after run {self.n}: {res['n_tried']} single changes screened; fitting the present network once more")
        kf0 = self._ladder(self.S, ph6, res["current"]["theta"]) if net and not self._out_of_time() else None
        if kf0 is not None and self._evidence(kf0) < bic0 - 1.0:
            self.kin, bic0 = kf0, self._evidence(kf0)               # a better fit of the SAME network: use it
            self.misfit = m = self.explains(ph6)
            log.update(refit=True, miss_x_noise=m["x_noise"], verdict=m["verdict"])
            if not self._needs_search(m):
                self.revision_log.append(dict(log, outcome="kept", seconds=time.time() - t0, words=(
                    f"After run {self.n}: a better fit of the same reactions was found; with it, {m['words']}. "
                    "The reaction network is unchanged.")))
                return False
        # ---- the best single changes, each fitted with the advisor's own kinetic model classes
        cands, top = [], res["moves"][:self.REVISE_TOP]
        for i, mv in enumerate(top):
            if self._out_of_time():
                self._timed_out = i < len(top)
                break
            self._say(f"after run {self.n}: comparing the best changes like for like ({i + 1} of {len(top)}: {mv['what']})")
            S_new = self._build(mv["reactions"])
            kf = self._ladder(S_new, ph6, mv["fit"]["theta"])
            if kf is not None:
                cands.append((self._evidence(kf), mv, S_new, kf))
        cands.sort(key=lambda c: c[0])
        if cands:
            bic1, mv, S_new, kf = cands[0]
            key1 = D.network_key(D.parse_network(S_new))
            need = self.REVISE_RETURN if key1 in self._left else self.REVISE_GAIN
            log.update(best_change=mv["what"], gain=float(bic0 - bic1), needed=need)
        if (not cands or bic0 - bic1 < need) and self._timed_out:
            return False                                            # not every change was compared: _revise says so
        if not cands or bic0 - bic1 < need:
            self.revision_log.append(dict(log, outcome="kept", seconds=time.time() - t0, words=(
                f"After run {self.n}: {m['words']}, but no single change of the network is clearly better"
                + (f" (the best, to {mv['what']}, lowers the evidence score BIC by {bic0 - bic1:.1f}; {need:g} is needed)"
                   if cands else "")
                + f". The network is unchanged; the advisor checks again after run {self.n + self.REVISE_WAIT}.")))
            return False
        verb = {"add": "Adding", "remove": "Removing", "change": "Changing", "replace": "Replacing"}
        first, _, rest = mv["what"].partition(" ")
        also = [c for c in cands[1:] if c[0] - bic1 < D.INDISTINGUISHABLE]
        tail = (f" Another single change, to {also[0][1]['what']}, explains these runs about as well (BIC difference "
                f"{also[0][0] - bic1:.1f}): the data cannot tell the two apart yet, and the advisor will switch if later "
                "runs clearly favour it." if also else "")
        self._adopt(ph6, net, S_new, kf, m, bic0, bic1, need, mv["kind"], mv["what"],
                    f"{verb.get(first, first.capitalize())} {rest} reduces", tail, [c[1]["what"] for c in also], log, t0)
        return True

    def _adopt(self, ph6, net, S_new, kf, m, bic0, bic1, need, kind, change, lead, tail, also, log, t0):
        """Take network S_new (kinetic fit kf) in place of the present one and write the change down."""
        self._left[D.network_key(net)] = int(self.n)
        self.S, self.kin = S_new, kf
        self._basis = (self.X.copy(), np.asarray(self.gc, float).copy(), self.hours.copy())    # the runs this change rests on
        m1 = self.explains(ph6)
        after = D.network_text(S_new)
        words = (f"After run {self.n}: {_what_was_wrong(m)}. {lead} the typical miss from {m['x_noise']:.1f} to "
                 f"{m1['x_noise']:.1f} times the GC noise (evidence score BIC, the misfit plus a penalty for every fitted "
                 f"constant, lower is better: {bic0:.0f} -> {bic1:.0f}; a drop of {need:g} or more is taken as strong "
                 f"evidence). The advisor now works with: {'; '.join(after)}.")
        if not m1["within_allowance"]:
            words += (" This network still does not explain every reading (typical miss "
                      f"{m1['x_noise']:.1f} times the GC noise), so the advisor keeps checking.")
        words += tail
        self.revisions.append(dict(run=int(self.n), kind=kind, change=change, before=D.network_text(net), after=after,
                                   bic_before=float(bic0), bic_after=float(bic1), gain=float(bic0 - bic1), needed=float(need),
                                   miss_before=m["x_noise"], miss_after=m1["x_noise"], fits_after=bool(m1["within_allowance"]),
                                   also_possible=list(also), words=words))
        self.revision_log.append(dict(log, outcome="revised", seconds=time.time() - t0, words=words))

    def _start_again(self, ph6, m):
        """The way out of a dead end.  Changing one reaction at a time can end at a network that
        does not explain the runs and that no single change improves (seen when the advisor is
        given a very different network: two wrong reactions can prop each other up).  The cure
        is the one the reaction search itself uses: search again from NO reactions at all
        (discovery.discover on the campaign's own runs, limited to REVISE_FRESH_SECONDS), and
        take the result if it beats the present network on the same evidence rule.  Only from
        REVISE_FRESH_RUNS runs on (5 screening runs cannot settle a whole network) and at most
        REVISE_FRESH_MAX times.  True if the network was changed."""
        if self.n < self.REVISE_FRESH_RUNS or self.n_fresh >= self.REVISE_FRESH_MAX:
            return False
        limit = self.REVISE_FRESH_SECONDS
        if self.revise_seconds is not None:
            limit = min(limit, self.revise_seconds - (time.time() - self._t_rev))
            if limit < 5.0:                                 # not enough time left for a search worth the name
                self._timed_out = True
                return False
        t0 = time.time()
        self.n_fresh += 1
        net = D.parse_network(self.S)
        self._say(f"after run {self.n}: no single change helps; searching the reactions afresh from none at all")
        res = D.discover(D.make_data(ph6, self.gc, self.hours, self.a_spec), library=self._library(),
                         max_reactions=self.MAX_REACTIONS, n_restarts=1, max_seconds=limit, seed=self.seed,
                         progress=(lambda info: self._say("fresh search: " + info["message"])) if self.progress else None)
        best = res["best"]
        log = dict(run=int(self.n), network=D.network_text(net), miss_x_noise=m["x_noise"], verdict=m["verdict"],
                   n_tried=res["n_fits"], fresh=True, best_change="start again from no reactions: " + "; ".join(best["reactions"]))
        kf = None
        if best["reactions"] and D.network_key(D.parse_network(best["reactions"])) != D.network_key(net):
            S_new = self._build(best["reactions"])
            kf = self._ladder(S_new, ph6, best["theta"])
        if kf is not None:
            bic0, bic1 = self._evidence(self.kin), self._evidence(kf)
            need = self.REVISE_RETURN if D.network_key(D.parse_network(S_new)) in self._left else self.REVISE_GAIN
            log.update(gain=float(bic0 - bic1), needed=need)
        if kf is None or bic0 - bic1 < need:
            self.revision_log.append(dict(log, outcome="kept", seconds=time.time() - t0, words=(
                f"After run {self.n}: {m['words']}, and a fresh search from no reactions at all found nothing clearly better. "
                "The network is unchanged.")))
            return False
        self._adopt(ph6, net, S_new, kf, m, bic0, bic1, need, "search", "start again from no reactions",
                    "No single change of the network was clearly better, so the reactions were searched afresh from none at "
                    "all (smart/discovery.py); the network found reduces", "", [], log, t0)
        return True

    def _physics(self, Xc, thetas, W=None, kpis=False, t=None):
        """Profit predicted by the physics model for every (theta, candidate) pair: (m, n).
        With kpis=True also returns conversion, selectivity and adiabatic rise, (m, n) each.
        t = time-on-stream of the candidates (default: the next run's)."""
        m, n = len(thetas), len(Xc)
        ph = np.tile(unscale(Xc), (m, 1))
        P = ph[:, 0]
        Wr = np.tile(np.broadcast_to(np.asarray(self.wcat if W is None else W, float), (n,)), m)
        tt = np.tile(np.broadcast_to(np.asarray(self.t_next if t is None else t, float), (n,)), m)
        F, F0 = self.kin.simulate(np.column_stack([ph, Wr]), np.repeat(thetas, n, 0), t=tt)
        with np.errstate(invalid="ignore"):
            pr = self.S.profit(F, F0, P, Wr).reshape(m, n)
        pr[np.isinf(pr)] = np.nan                      # NaN = this chemistry gave no answer here
        if not kpis:
            return pr
        with np.errstate(invalid="ignore"):
            XA, SC, _ = self.S.kpis(F, F0)
            dT = self.S.dT_ad(F, F0)
        return pr, XA.reshape(m, n), SC.reshape(m, n), dT.reshape(m, n)

    def _predict(self, Xc, t=None, include_noise=False, thetas=None):
        Xc = np.atleast_2d(Xc)
        pr = self._physics(Xc, self.thetas if thetas is None else thetas, t=t)
        mu_p = np.nan_to_num(pr[0], nan=-1e3)          # best-estimate chemistry
        with np.errstate(invalid="ignore"):
            sd_p = np.nan_to_num(np.nanstd(pr, 0), nan=0.0)   # disagreement between plausible chemistries
        mu_r, sd_r, extra = self._correct(Xc, mu_p, include_noise)
        return (mu_p + mu_r, np.sqrt(sd_p ** 2 + sd_r ** 2 + 1e-9),
                dict(mu_phys=mu_p, sd_phys=sd_p, mu_corr=mu_r, sd_corr=sd_r, **extra))

    def _correct(self, Xc, mu_phys, include_noise=False):
        """The correction to the physics at Xc: mean, sd, and (v6) the committee's parts."""
        if self.committee is not None and "direct" in self.committee.active:
            # v7: where the physics gave no answer (the -1e3 penalty) the stand-alone network must not overrule it
            return self.committee.predict(Xc, mu_phys, include_noise=include_noise, phys_ok=np.asarray(mu_phys) > -999.0)
        if self.committee is not None:
            mu, sd, extra = self.committee.predict(Xc, mu_phys, include_noise=include_noise)
            return mu, sd, extra
        if self.gp is not None:
            mu, sd = self.gp.predict(Xc, include_noise=include_noise)
            return mu, sd, {}
        return np.zeros(len(Xc)), np.zeros(len(Xc)), {}

    def predict(self, Xc, parts=False, t=None):
        mu, sd, p = self._predict(Xc, t=t)
        sd = sd * self.calib
        return (mu, sd, p) if parts else (mu, sd)

    def feasibility(self, Xc, t=None):
        """P(meets the spec) over the plausible chemistries, and the mean KPIs: (n,), dict."""
        Xc = np.atleast_2d(Xc)
        pr, XA, SC, dT = self._physics(Xc, self.thetas, kpis=True, t=t)
        ok = np.ones_like(pr, bool)
        for key, lim in self.spec.items():
            val = {"S_C": SC, "X_A": XA, "dT_ad": dT}[key]
            ok &= (val >= lim) if SPEC_KEYS[key][0] == "min" else (val <= lim)
        ok &= np.isfinite(pr)
        with np.errstate(invalid="ignore"):
            kp = dict(X_A=np.nanmean(XA, 0), S_C=np.nanmean(SC, 0), dT_ad=np.nanmean(dT, 0))
        return ok.mean(0), kp

    def recommend(self):
        """Index of the tested run with the best PREDICTED profit (a single lucky GC reading
        cannot fool it) among the runs that met the spec, measured; falls back to all runs."""
        mu = self.mu_data
        if self.spec:
            # measured KPIs: from the GC reading, with the network the advisor was given (S_start is S
            # unless revise=True changed the network), so a revision never changes what was measured
            ph6 = physical(self.X, self.W)
            XA, SC, _ = kpis_from_gc(self.S_start, ph6, self.gc)
            dT = dT_ad_from_gc(self.S_start, ph6, self.gc)
            ok = np.ones(self.n, bool)
            for key, lim in self.spec.items():
                val = {"S_C": SC, "X_A": XA, "dT_ad": dT}[key]
                ok &= (val >= lim) if SPEC_KEYS[key][0] == "min" else (val <= lim)
            if ok.any():
                idx = np.nonzero(ok)[0]
                return int(idx[np.argmax(mu[idx])])
        return int(np.argmax(mu))

    # ------------------------------------------------------------------ planning
    def _pin(self, C):
        C = np.clip(C, self.lo, self.hi)
        for i, v in self.fixed.items():
            C[:, i] = v
        return C

    def _ok(self, C):
        """Feasible (flow minimum) and, with a spec, likely to meet it."""
        keep = feasible(unscale(C))
        if self.spec and keep.any():
            pf = np.zeros(len(C))
            pf[keep] = self.feasibility(C[keep])[0]
            keep &= pf >= 0.5
        return keep

    def plan(self, rng=None, seeds=None):
        """Where the model expects the best profit over the WHOLE safety box (not just one
        step away): x*, its predicted profit and uncertainty.
        seeds: optional settings (scaled, n x 5) to consider as well, e.g. a point the operator
        picked on the profit-against-selectivity curve (smart/pareto.py).  The search is a random
        one, and under a product spec it can miss a good allowed point; a seed that the spec rule
        accepts is never missed, and the local refinement starts from it if it is among the best.
        Without seeds the search, and the random numbers it uses, are exactly as before."""
        rng = rng or self.rng
        x_best = self.X[int(np.argmax(self.mu_data))]
        C1 = np.vstack([sobol_in(self.lo, self.hi, 1024, rng, self.free), self.X,
                        x_best + rng.normal(0, 0.5, (64, DIM)) * self.step])
        if seeds is not None and len(seeds):
            C1 = np.vstack([C1, np.atleast_2d(np.asarray(seeds, float))])
        C1 = self._pin(C1)
        keep = self._ok(C1)
        if not keep.any():                      # nothing meets the spec anywhere: plan without it
            keep = feasible(unscale(C1))
            if not keep.any():
                return x_best.copy(), float(self.mu_data.max()), 0.0
        C1 = C1[keep]
        mu1 = self.predict(C1)[0]
        top = C1[np.argsort(mu1)[-5:]]
        C2 = self._pin(top[rng.integers(len(top), size=256)] + rng.normal(0, 0.03, (256, DIM)))
        C2 = C2[self._ok(C2)]
        mu2, sd2 = self.predict(C2) if len(C2) else (np.array([]), np.array([]))
        C, mu, sd = np.vstack([C1, C2]), np.r_[mu1, mu2], np.r_[np.full(len(C1), np.nan), sd2]
        k = int(np.argmax(mu))
        if np.isnan(sd[k]):
            sd[k] = self.predict(C[k:k + 1])[1][0]
        return C[k], float(mu[k]), float(sd[k])

    def steps_away(self, x_from, x_to):
        """How many runs it takes to get from x_from to x_to under the step limits."""
        d = np.abs(np.asarray(x_to) - np.asarray(x_from))
        return int(np.ceil(np.max(d / np.where(self.step > 0, self.step, 1.0) - 1e-9)))

    def _step_candidates(self, x_from, x_star, rng, n_sobol):
        """Structured candidates for one allowed step from x_from: the step toward x*, the box
        corners (full steps on every free knob), points on the line to x*, stay, Sobol."""
        blo, bhi = step_box(x_from, self.step, self.lo, self.hi, self.fixed)
        pts = [np.clip(x_star, blo, bhi), x_from.copy()]
        for f in (0.25, 0.5, 0.75):
            pts.append(np.clip(x_from + f * (x_star - x_from), blo, bhi))
        nf = len(self.free)
        if nf:
            corners = np.array(np.meshgrid(*[[0, 1]] * nf)).reshape(nf, -1).T
            for c in corners:
                p = x_from.copy()
                p[self.free] = np.where(c == 1, bhi[self.free], blo[self.free])
                pts.append(p)
        C = np.vstack([np.array(pts), sobol_in(blo, bhi, n_sobol, rng, self.free)])
        return self._pin(C)

    def suggest(self, x_prev=None, rng=None, seeds=None):
        """The next run, reachable from x_prev (default: the last logged run) in one step, and why.
        seeds: optional settings (scaled, n x 5) the plan must consider as targets (see plan).
        With revise=True the info also holds the reaction network in use ('network', text lines),
        the changes made to it so far ('revisions', plain words under 'words') and the last
        check of the network against the GC readings ('network_check')."""
        x, info = self._suggest(x_prev, rng, seeds)
        if self.revise:
            info.update(network=self.network(), revisions=list(self.revisions),
                        network_check=None if self.misfit is None else self.misfit["words"])
        return x, info

    def _suggest(self, x_prev=None, rng=None, seeds=None):
        rng = rng or self.rng
        x_prev = np.asarray(self.X[-1] if x_prev is None else x_prev, float)
        if self.planner == "greedy":
            return self._suggest_greedy(x_prev, rng, seeds)
        x_star, mu_star, sd_star = self.plan(rng, seeds)
        best = float(self.mu_data.max())
        H = max(1, min(self.horizon, self.remaining))
        R = max(self.remaining, H)
        # ---- first-step candidates
        C1 = self._step_candidates(x_prev, x_star, rng, 256)
        keep = self._ok(C1)
        if not keep.any():
            # nothing within one step is likely to meet the spec (or the flow minimum): take the
            # step that gets closest to meeting it, heading toward the model optimum
            flow_ok = feasible(unscale(C1))
            if not flow_ok.any():
                return x_prev.copy(), dict(pred=best, sd=0.0, ei=0.0, best_pred=best, model_opt=x_star,
                                           model_opt_pred=mu_star, model_opt_sd=sd_star,
                                           steps_away=self.steps_away(x_prev, x_star), toward_opt=False,
                                           mode="stay", score=best, p_spec=0.0)
            Cf = C1[flow_ok]
            pf = self.feasibility(Cf)[0] if self.spec else np.ones(len(Cf))
            dist = np.max(np.abs(Cf - x_star) / np.where(self.step > 0, self.step, 1.0), axis=1)
            j = int(np.lexsort((dist, -np.round(pf, 2)))[0])            # most likely to meet the spec, then closest to x*
            x = Cf[j]
            mu, sd = self.predict(x[None])
            return x, dict(pred=float(mu[0]), sd=float(sd[0]), ei=0.0, best_pred=best, model_opt=x_star,
                           model_opt_pred=mu_star, model_opt_sd=sd_star,
                           steps_away=self.steps_away(x_prev, x_star), toward_opt=True,
                           mode="recover", score=float(mu[0]), p_spec=float(pf[j]))
        C1 = C1[keep]
        mu1, sd1 = self.predict(C1)
        ei1 = expected_improvement(mu1, sd1, best)
        pick = set(np.argsort(ei1)[-6:]) | set(np.argsort(mu1)[-3:]) | {0, 1}   # 0 = toward x*, 1 = stay
        M = sorted(pick)
        # dedupe within a quarter step
        chosen = []
        for i in M:
            if all(np.max(np.abs(C1[i] - C1[j]) / np.where(self.step > 0, self.step, 1)) > 0.25 for j in chosen):
                chosen.append(i)
        M = chosen
        # ---- greedy rollout on the model mean, batched per depth over all first steps
        traj = [np.array([C1[i] for i in M])]                                 # (m, DIM) per depth
        for h in range(1, H):
            cand = [self._step_candidates(x, x_star, rng, 24) for x in traj[-1]]
            sizes = [len(c) for c in cand]
            allc = np.vstack(cand)
            ok = self._ok(allc)
            mu = self._predict(allc, thetas=self.thetas[:1])[0]
            mu[~ok] = -np.inf
            nxt, start = [], 0
            for x, sz in zip(traj[-1], sizes):
                seg = mu[start:start + sz]
                nxt.append(allc[start + int(np.argmax(seg))] if np.isfinite(seg.max()) else x)
                start += sz
            traj.append(np.array(nxt))
        T = np.stack(traj, 1)                                                   # (m, H, DIM)
        # ---- score every trajectory under every plausible chemistry (+ GP draws)
        flat = T.reshape(-1, DIM)
        pr = self._physics(flat, self.thetas)                                   # (n_theta, m*H)
        pr = np.nan_to_num(pr, nan=-1e3)
        mu_r, sd_r, _ = self._correct(flat, pr[0])
        z = rng.standard_normal((self.n_theta, 1))
        f = pr + mu_r[None, :] + z * (sd_r * self.calib)[None, :]
        f = f.reshape(self.n_theta, len(M), H)
        ret = f.sum(2) + (R - H) * np.maximum(best, f.max(2))                   # (n_theta, m)
        score = ret.mean(0) / R                                                 # average profit per remaining run
        j = int(np.argmax(score))
        x = C1[M[j]]
        i1 = M[j]
        d_opt, d_step = x_star - x_prev, x - x_prev
        toward = bool(np.dot(d_opt, d_step) > 0.5 * np.linalg.norm(d_opt) * np.linalg.norm(d_step) + 1e-12)
        if np.max(np.abs(x - x_prev)) < 1e-9:
            mode = "stay"
        elif mu1[i1] - best > 1.64 * sd1[i1]:
            mode = "improve"
        elif toward and mu1[i1] < best:
            mode = "walk"
        else:
            mode = "check"
        pf, kp = (self.feasibility(x[None]) if self.spec else (np.array([1.0]), {}))
        return x, dict(pred=float(mu1[i1]), sd=float(sd1[i1]), ei=float(ei1[i1]), best_pred=best,
                       model_opt=x_star, model_opt_pred=mu_star, model_opt_sd=sd_star,
                       steps_away=self.steps_away(x_prev, x_star), toward_opt=toward, mode=mode,
                       score=float(score[j]), p_spec=float(pf[0]), trajectory=T[j],
                       traj_profit=[float(v) for v in f[0, j]])

    def _suggest_greedy(self, x_prev, rng, seeds=None):
        """The v4 rule: Expected Improvement one step ahead, and a purposeful step on the
        bottleneck knob toward x* when nothing within reach beats the best run."""
        x_star, mu_star, sd_star = self.plan(rng, seeds)
        blo, bhi = step_box(x_prev, self.step, self.lo, self.hi, self.fixed)
        waypoint = np.clip(x_star, blo, bhi)
        best = float(self.mu_data.max())
        x_best = self.X[int(np.argmax(self.mu_data))]
        C = np.vstack([sobol_in(blo, bhi, 512, rng, self.free), waypoint[None],
                       np.clip(waypoint + rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi),
                       np.clip(x_best + rng.normal(0, 0.3, (64, DIM)) * self.step, blo, bhi)])
        C = self._pin(C)
        ok = self._ok(C)
        if not ok.any():
            return x_prev.copy(), dict(pred=best, sd=0.0, ei=0.0, best_pred=best, model_opt=x_star,
                                       model_opt_pred=mu_star, model_opt_sd=sd_star,
                                       steps_away=self.steps_away(x_prev, x_star), toward_opt=False, mode="stay")
        C = C[ok]
        mu, sd = self.predict(C)
        ei = expected_improvement(mu, sd, best)
        j = int(np.argmax(ei))
        mode = "improve"
        far_gain = mu_star - 1.64 * sd_star - best
        if far_gain > 0 and ei[j] < 0.1 * far_gain:
            dist = np.abs(x_star - x_prev) / np.where(self.step > 0, self.step, 1.0)
            k = int(np.argmax(dist))
            progress = np.abs(x_star[k] - C[:, k]) <= np.abs(x_star[k] - x_prev[k]) - 0.9 * self.step[k]
            if progress.any():
                cand = np.nonzero(progress)[0]
                j = int(cand[np.argmax(mu[cand])])
            else:
                j = int(np.argmin(np.max(np.abs(C - waypoint), axis=1)))
            mode = "walk"
        x = C[j]
        d_opt, d_step = x_star - x_prev, x - x_prev
        toward = bool(np.dot(d_opt, d_step) > 0.5 * np.linalg.norm(d_opt) * np.linalg.norm(d_step) + 1e-12)
        return x, dict(pred=float(mu[j]), sd=float(sd[j]), ei=float(ei[j]), best_pred=best,
                       model_opt=x_star, model_opt_pred=mu_star, model_opt_sd=sd_star,
                       steps_away=self.steps_away(x_prev, x_star), toward_opt=toward,
                       waypoint=waypoint, mode=mode)


# =================================================================================== campaigns
def _new_history():
    return dict(X=[], gc=[], y_obs=[], y_true=[], X_A=[], S_C=[], dT_ad=[], hours=[], activity=[], rec=[], check=[])


def _evaluate(vr, h, x, check=False):
    r = vr.run(x)
    h["X"].append(np.asarray(x, float).copy()); h["gc"].append(r["gc"])
    h["y_obs"].append(r["profit"]); h["y_true"].append(r["profit_true"])
    h["X_A"].append(r["X_A"]); h["S_C"].append(r["S_C"]); h["dT_ad"].append(r["dT_ad"])
    h["hours"].append(r["hours"]); h["activity"].append(r["activity"]); h["check"].append(bool(check))
    h["rec"].append(h["y_true"][int(np.argmax(h["y_obs"]))])
    return r


def _score_rec(vr, h, idx):
    """True profit, at the CURRENT catalyst activity, of the settings the method recommends."""
    h["rec"][-1] = vr.true_profit(h["X"][idx])


def start_campaign(system, seed=0, wcat=WCAT, a_spec=A_SPEC, deactivation=0.0):
    """The virtual reactor and the operator's 5 screening runs (shared by every method)."""
    S = SYSTEMS[system]
    vr = VirtualReactor(S, seed, wcat, a_spec, deactivation=deactivation)
    h = _new_history()
    base = base_case(seed)
    for row in screening_runs(base):
        _evaluate(vr, h, scale(row)[0])
    return S, vr, h, base


def is_check_run(n, check_every):
    """Protocol: after the 5 screening runs, repeat the base case every `check_every` runs."""
    return bool(check_every) and n >= 5 and (n - 5) % check_every == check_every - 1


def _pack(h, **extra):
    out = {k: np.array(v) for k, v in h.items()}
    out["cum_true"] = np.cumsum(out["y_true"])
    out.update(extra)
    return out


def model_system(model, S):
    """The reaction network the advisor is GIVEN for a campaign on the true system S.
    None                 the right network (S itself; still without the reactant adsorption)
    a name               systems.MODELS ('complex_missing_E': the over-reaction left out), or one of
                         the candidate networks of smart/discovery.py: 'cand_close' (one reaction
                         order wrong), 'cand_similar' (another pathway, no equilibrium), 'cand_far'
                         (two reactions, other orders, nothing makes E), 'cand_truth'
    a ReactionSystem     any network, e.g. discovery.build_system("A + B -> C\\nA -> D") or the
                         network a reaction search returned
    Returns (system, name used in the campaign's strategy label)."""
    if model is None:
        return S, ""
    if isinstance(model, str):
        if model in MODELS:
            M = MODELS[model]
        elif model.startswith("cand_") and model[5:] in D.CANDIDATES:
            M = D.candidate_system(model[5:])
        else:
            raise ValueError(f"unknown model {model!r}: use one of {', '.join(MODELS)}, "
                             f"{', '.join('cand_' + k for k in D.CANDIDATES)}, or a ReactionSystem")
        name = model
    else:
        M, name = model, str(getattr(model, "name", "network")).replace(" ", "_")
    if list(M.species) != list(S.species):
        raise ValueError(f"the model's species ({', '.join(M.species)}) are not those of the '{S.name}' system "
                         f"({', '.join(S.species)}), so it cannot be compared with that system's GC readings")
    return M, name


def run_smart(system, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, verbose=False, model=None, callback=None,
              planner="horizon", models=None, deactivation=0.0, check_every=0, spec=None, correction="gp",
              members=None, revise=False, library=None):
    """One campaign with the smart advisor on the virtual reactor.
    model: the reaction network the advisor is given, when it is not the right one: a name (see
    model_system: 'complex_missing_E', 'cand_close', 'cand_similar', 'cand_far') or a ReactionSystem
    object; default = the right network (still without the reactant adsorption).
    planner / models: "horizon" + all kinetic models = v5; "greedy" + ("plain",) = v4;
    correction="committee" with the v5 planner = v6; correction="hybrid" adds a stand-alone neural
    network on the profit.  members: committee members to use (None = the default of the
    correction), e.g. ("none", "gp") = v6 without its neural networks.
    revise=True: the advisor may change its reaction network during the campaign when the network
    does not explain the GC readings (SmartAdvisor._revise); library = the reactions it may use.
    The v7 advisor is correction="hybrid" with revise=True (strategy 'smart_v7'); hybrid with
    revise=False is labelled 'smart_v7_fixednet'.
    Logged weights: v6 with its default members keeps committee_weights as (updates, 3), none / gp /
    nn.  Every other committee logs (updates, 4) in the order of weight_columns (none / gp / nn /
    direct, 0 for a member that is absent or has not joined yet), plus committee_members (the
    members used), committee_weights_final (after the last run) and committee_heldout_rmse
    (updates, 4): how far each member was from runs it had not seen (NaN if absent).
    Logged network (for v7, for revise=True and for a model given as an object or as 'cand_...'; the
    v5 / v6 campaigns keep exactly their old keys): network (one text per run: the reactions the
    advisor held when that run was chosen, one reaction per line, in the text format that
    discovery.parse_network reads; the 5 screening runs carry the starting network),
    network_start, network_final (after the last run), n_revisions, n_searches, revision_runs
    (number of runs done when each change was made), revision_words (each change in plain words),
    revisions_json (the full records as JSON text) and network_fit (updates,): the typical miss
    of the advisor's kinetic fit at each update, as a multiple of the GC noise.
    Every campaign of the hybrid committee or of revise=True also logs v7_version (V7_VERSION), so
    that benchmark.is_stale can tell files made by an earlier v7 from the present one.
    callback(run_number, x, result, info) is called after every advisor-chosen run."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec, deactivation)
    x_base = scale(base)[0]
    M, model_name = model_system(model, S)
    adv = SmartAdvisor(M, wcat=wcat, a_spec=a_spec, seed=seed, planner=planner, models=models, spec=spec,
                       remaining=budget - 5, correction=correction, members=members, revise=revise, library=library)
    legacy = correction == "committee" and adv.members == MEMBERS      # the v6 log format, read by compare_v6.py
    log_net = revise or correction == "hybrid" or not (model is None or (isinstance(model, str) and model in MODELS))
    rng = np.random.default_rng(seed)
    info_log, w_log, e_log = [], [], []
    net_log, fit_log = ["\n".join(adv.network())] * len(h["X"]) if log_net else [], []
    while len(h["X"]) < budget:
        adv.remaining = budget - len(h["X"])
        adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]), hours=np.array(h["hours"]))
        if log_net:
            net_log.append("\n".join(adv.network()))
            fit_log.append((adv.misfit if adv.revise and adv.misfit else adv.explains())["x_noise"])
        if adv.committee is not None and legacy:
            w_log.append([adv.committee.weights.get(k, 0.0) for k in ("none", "gp", "nn")])
        elif adv.committee is not None:
            w_log.append(adv.committee.weight_row())
            e_log.append(adv.committee.weight_row("heldout_rmse", np.nan))
        _score_rec(vr, h, adv.recommend())
        n = len(h["X"])
        if is_check_run(n, check_every):
            x, info, chk = x_base.copy(), dict(mode="check-run", pred=float("nan"), sd=0.0), True
        else:
            x, info = adv.suggest(h["X"][-1], rng)
            chk = False
        r = _evaluate(vr, h, x, check=chk)
        info_log.append(info)
        if len(h["X"]) == budget:
            adv.update(np.array(h["X"]), np.array(h["gc"]), np.array(h["y_obs"]), hours=np.array(h["hours"]))
        _score_rec(vr, h, adv.recommend())
        if callback:
            callback(len(h["X"]), x, r, info)
        if verbose:
            print(f"smart {system} run {len(h['X']):2d}: measured {r['profit']:7.2f}  predicted "
                  f"{info.get('pred', float('nan')):7.2f} +- {info.get('sd', 0):.2f}   recommended (true) {h['rec'][-1]:7.2f}")
    name = "smart" if planner == "horizon" and models is None else "smart_v4"
    if correction == "committee":
        name = "smart_v6"
    elif correction == "hybrid":
        name = "smart_v7" if revise else "smart_v7_fixednet"
    if revise and correction != "hybrid":
        name += "_revise"
    if model_name:
        name += f"_{model_name}"
    extra = {}
    if adv.committee is not None and legacy:
        extra["committee_weights"] = np.array(w_log)          # (updates, 3): none / gp / nn
    elif adv.committee is not None:
        extra.update(committee_weights=np.array(w_log),       # (updates, 4): none / gp / nn / direct
                     weight_columns=np.array(ALL_MEMBERS), committee_members=np.array(adv.members),
                     committee_weights_final=np.array(adv.committee.weight_row()),
                     committee_heldout_rmse=np.array(e_log))
    if log_net:
        extra.update(network=np.array(net_log), network_start=net_log[0], network_final="\n".join(adv.network()),
                     network_fit=np.array(fit_log), n_revisions=len(adv.revisions),
                     revision_runs=np.array([r["run"] for r in adv.revisions], int),
                     revision_words=np.array([r["words"] for r in adv.revisions], str),
                     revisions_json=json.dumps(adv.revisions), revisions=list(adv.revisions),
                     n_searches=int(adv.n_searches))
    if correction == "hybrid" or revise:
        extra["v7_version"] = V7_VERSION
    return _pack(h, strategy=name, system=system, seed=seed, base=base, theta=adv.kin.theta,
                 kin_model=adv.kin.model, info=info_log, advisor=adv, **extra)


# ----------------------------------------------------------------------------------- baselines
class ChemistPolicy:
    """A scripted rules-of-thumb policy (a fixed script, not a person): read conversion and
    selectivity off the GC, apply rules of thumb, and keep whatever worked.

      selectivity poor (< 0.85): cooler (side reactions have the higher activation energy),
                                 compensating with more pressure; then more B; then dilute
      conversion poor  (< 0.70): hotter and more pressure; then less flow (longer contact);
                                 then less inert
      both fine:                 push throughput (more A and B, with pressure); then hotter;
                                 then less inert

    Momentum: if the last change improved on the best run, repeat it.  Otherwise go back to
    the best run and try the next rule.  When a new best appears the rule list restarts.
    A move is one full step of each knob involved, clipped to the step limits from the
    previous run, and a chemist never repeats an experiment: a move that lands on a point
    already run is skipped.  When every rule has been tried from the current best, the
    operator takes a random allowed step ("try something").  Uses only what a person can
    see: the GC reading and the profit."""
    S_LOW, X_LOW = 0.85, 0.70
    KNOB = {"T": 4, "P": 0, "A": 1, "B": 2, "I": 3}

    def __init__(self, step, lo, hi, fixed=None, seed=0):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 31)
        self.last, self.best_idx, self.k = None, -1, 0

    def rules(self, XA, SC):
        if np.isnan(SC) or SC < self.S_LOW:
            return ["T- P+", "T-", "P+", "B+", "I+", "T- P+ B+"]
        if XA < self.X_LOW:
            return ["T+ P+", "P+", "T+", "A- B-", "I-", "T+ P+ I-"]
        return ["A+ B+", "A+ B+ P+", "P+", "T+", "I-", "A+ B+ T+"]

    def _apply(self, origin, name, x_last, X):
        x = origin.copy()
        for tok in name.split():
            i, d = self.KNOB[tok[0]], (+1 if tok[1] == "+" else -1)
            if i in self.fixed:
                return None
            x[i] = origin[i] + d * self.step[i]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        x = np.clip(x, blo, bhi)
        if not feasible(unscale(x))[0]:
            return None
        if any(np.max(np.abs(x - xr)) < 1e-3 for xr in X):          # already run: pointless
            return None
        return x

    def next(self, h):
        X, Y = h["X"], h["y_obs"]
        ib = int(np.argmax(Y))
        x_last = X[-1]
        if ib == len(X) - 1 and self.last is not None:            # last change worked: do it again
            x = self._apply(x_last, self.last, x_last, X)
            if x is not None:
                return x
        if ib != self.best_idx:
            self.best_idx, self.k = ib, 0
        lst = self.rules(h["X_A"][ib], h["S_C"][ib])
        for _ in range(len(lst)):
            name = lst[self.k % len(lst)]
            self.k += 1
            x = self._apply(X[ib], name, x_last, X)
            if x is not None:
                self.last = name
                return x
        self.last = None                                           # every rule tried: try something
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        for _ in range(50):
            x = blo + self.rng.random(DIM) * (bhi - blo)
            if feasible(unscale(x))[0]:
                return x
        return x_last.copy()

    def recommend(self, h):
        return int(np.argmax(h["y_obs"]))


class PlainBOPolicy:
    """Black-box Bayesian optimisation under the same step limits: the fitted GP of v2
    (Matern, ARD, log P, rank output transform) + Expected Improvement, but it sees only the
    profit numbers and may only propose runs within one step of the previous run."""

    def __init__(self, step, lo, hi, seed=0, fixed=None):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 11)
        self.seed = seed
        self.gp = None

    def _fit(self, h):
        X, Y = np.array(h["X"]), np.array(h["y_obs"])
        self.gp = LogInputs(GP(seed=self.seed), (0,), LOW, HIGH).fit(X, copula_transform(Y))
        self.X = X

    def next(self, h):
        self._fit(h)
        X = self.X
        x_last = X[-1]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        x_best = X[int(np.argmax(self.gp.predict(X)[0]))]
        C = np.vstack([sobol_in(blo, bhi, 512, self.rng),
                       np.clip(x_best + self.rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi)])
        for i, v in self.fixed.items():
            C[:, i] = v
        C = C[feasible(unscale(C))]
        if len(C) == 0:
            return x_last.copy()
        mu, sd = self.gp.predict(C)
        best = float(np.max(self.gp.predict(X)[0]))
        return C[int(np.argmax(expected_improvement(mu, sd, best)))]

    def recommend(self, h):
        if self.gp is None or len(self.X) != len(h["X"]):
            self._fit(h)
        return int(np.argmax(self.gp.predict(self.X)[0]))


class NNOnlyPolicy:
    """A neural network alone, under the same step limits: the candidates and the recommendation
    rule of PlainBOPolicy, but the model is a deep ensemble of small networks (knobs -> profit,
    committee.ProfitEnsemble) and Expected Improvement uses the ensemble's mean and the spread
    between its networks.  No physics and no GC composition: like plain BO it sees one profit
    number per run.  It shows how a neural network on its own chooses runs.
    The recommendation is the tested run with the best MEASURED profit, the rule of the
    rules-of-thumb policy and of the random walk.  An earlier version recommended the run with
    the highest ensemble mean, as plain BO does with its GP; but the ensemble does not reproduce
    its own training runs (bootstrap and weight penalty: a run measured at +0.9 per hour was
    rated below one measured at -11.7), so it recommended runs that had measured far worse than
    its best one, and its score showed that rule more than the network's choice of runs.  Plain
    BO's GP reproduces the measured ranking, so for it the two rules hardly differ.
    Campaign files carry recommend_rule = 'best measured run' (benchmark.is_stale flags files
    made under the old rule)."""
    RECOMMEND_RULE = "best measured run"

    def __init__(self, step, lo, hi, seed=0, fixed=None):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 37)
        self.seed = seed
        self.nn = None

    def _fit(self, h):
        X, Y = np.array(h["X"]), np.array(h["y_obs"])
        self.nn = ProfitEnsemble(self.seed).fit(X, Y)
        self.X = X
        self.mu_X = self.nn.predict(X)[0]

    def next(self, h):
        """The next run (scaled knobs), within one allowed step of the last run in the history h:
        the candidate with the highest Expected Improvement under the ensemble (mean and spread),
        measured against the ensemble's own best mean at the runs done."""
        self._fit(h)
        X = self.X
        x_last = X[-1]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        x_best = X[int(np.argmax(self.mu_X))]
        C = np.vstack([sobol_in(blo, bhi, 512, self.rng),
                       np.clip(x_best + self.rng.normal(0, 0.3, (128, DIM)) * self.step, blo, bhi)])
        for i, v in self.fixed.items():
            C[:, i] = v
        C = C[feasible(unscale(C))]
        if len(C) == 0:
            return x_last.copy()
        mu, sd = self.nn.predict(C)
        return C[int(np.argmax(expected_improvement(mu, sd, float(np.max(self.mu_X)))))]

    def recommend(self, h):
        """Index of the tested run with the best measured profit (see the class note)."""
        return int(np.argmax(h["y_obs"]))


class RandomWalkPolicy:
    """Trial and error with no rules at all: a random allowed step from the previous run."""

    def __init__(self, step, lo, hi, seed=0, fixed=None):
        self.step, self.lo, self.hi, self.fixed = step, lo, hi, dict(fixed or {})
        self.rng = np.random.default_rng(seed + 23)

    def next(self, h):
        x_last = h["X"][-1]
        blo, bhi = step_box(x_last, self.step, self.lo, self.hi, self.fixed)
        for _ in range(50):
            x = blo + self.rng.random(DIM) * (bhi - blo)
            if feasible(unscale(x))[0]:
                return x
        return x_last.copy()

    def recommend(self, h):
        return int(np.argmax(h["y_obs"]))


POLICIES = {"chemist": ChemistPolicy, "plain_bo": PlainBOPolicy, "random_walk": RandomWalkPolicy,
            "nn_only": NNOnlyPolicy}


def run_baseline(system, method, budget=25, seed=0, wcat=WCAT, a_spec=A_SPEC, max_step=None,
                 deactivation=0.0, check_every=0):
    """One campaign with a baseline policy, starting from the same 5 screening runs."""
    S, vr, h, base = start_campaign(system, seed, wcat, a_spec, deactivation)
    x_base = scale(base)[0]
    step, lo, hi = step_scaled(max_step), np.zeros(DIM), np.ones(DIM)
    pol = POLICIES[method](step, lo, hi, seed=seed)
    while len(h["X"]) < budget:
        n = len(h["X"])
        if is_check_run(n, check_every):
            x, chk = x_base.copy(), True
        else:
            x, chk = pol.next(h), False
        _evaluate(vr, h, x, check=chk)
        _score_rec(vr, h, pol.recommend(h))
    extra = dict(recommend_rule=pol.RECOMMEND_RULE) if hasattr(pol, "RECOMMEND_RULE") else {}
    return _pack(h, strategy=method, system=system, seed=seed, base=base, **extra)


# =================================================================================== true optimum (for scoring only)
_OPT_FILE = os.path.join(os.path.dirname(__file__), "optima.json")


def true_optimum(system, wcat=WCAT, a_spec=A_SPEC, recompute=False):
    """Global optimum of the TRUE profit at this catalyst load and full activity (differential
    evolution on the fast simulator, then polished with the exact LSODA solver), subject to the
    flow constraint.  Only used to score methods; the advisor never sees it.  Cached in
    smart/optima.json."""
    key = f"{system}|W={float(wcat):g}|a={float(a_spec):g}"
    cache = {}
    if os.path.exists(_OPT_FILE):
        with open(_OPT_FILE) as fh:
            cache = json.load(fh)
    if key in cache and not recompute:
        return cache[key]
    S = SYSTEMS[system]

    def fast(Xc):
        ph = unscale(Xc)
        P, VA, VB, VI, T = ph.T
        F, F0 = S.simulate(P, VA, VB, VI, wcat, T, truth=True, a_spec=a_spec)
        pen = 1e3 * np.maximum(0.0, MIN_TOTAL_FLOW - (VA + VB + VI))
        return S.profit(F, F0, P, wcat) - pen

    vr = VirtualReactor(S, 0, wcat, a_spec)
    best = None
    for s in range(2):
        res = differential_evolution(lambda Z: -fast(Z.T), [(0, 1)] * DIM, vectorized=True,
                                     seed=s, popsize=30, maxiter=300, tol=1e-10, polish=False)
        if best is None or res.fun < best.fun:
            best = res

    def exact(z):
        z = np.clip(z, 0, 1)
        if not feasible(unscale(z))[0]:
            return 1e6
        return -vr.true_profit(z)

    pol = minimize(exact, best.x, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=600))
    x = np.clip(pol.x, 0, 1)
    f = vr.true_profit(x)
    Xr = np.random.default_rng(123).random((4000, DIM))
    Xr = Xr[feasible(unscale(Xr))]
    ys = fast(Xr)
    cache[key] = dict(f_opt=f, x_opt=x.tolist(), x_opt_phys=unscale(x)[0].tolist(),
                      wcat=float(wcat), a_spec=float(a_spec),
                      f_median=float(np.median(ys)), frac_profitable=float(np.mean(ys > 0)))
    tmp = _OPT_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp, _OPT_FILE)
    return cache[key]
