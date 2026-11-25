"""
REACTOR ADVISOR - the web interface.

    python reactor_ui.py                      # opens http://localhost:8765 in your browser
    python reactor_ui.py --port 9000 --no-browser

Everything the command-line tool (reactor_advisor.py) does, in a page a chemist can use
without a terminal: set up the campaign (chemistry, catalyst, step limits, safety ranges,
economics), type in or import runs, get the next run with the reasons, see what the model
learned, explore what the model expects when a knob moves, and try it all on the virtual
reactor first.  Also (v7): the reaction finder (which reactions do the GC readings support?
smart/discovery.py, with worked example files), the v7 advisor that checks and repairs its
own reaction network, and an optional profit vs selectivity trade-off curve (smart/pareto.py).

No extra packages: Python's own http.server + the smart/ package.  Runs on your own machine
and nothing leaves it.  One campaign at a time (the page is the campaign).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import io
import csv
import copy
import json
import time
import argparse
import threading
import warnings
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart.systems import (SYSTEMS, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW, WCAT, A_SPEC, R, T_REF, DIM,
                           unscale, scale, physical, feasible, base_case, screening_runs, theta_to_physical)
from smart.advisor import SmartAdvisor, VirtualReactor, profit_from_gc, kpis_from_gc, dT_ad_from_gc, step_box
from smart.kinetics import HOURS_PER_RUN
from smart import discovery as D
from smart.pareto import tradeoff as pareto_tradeoff, spec_for_point

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
EX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "discovery")
VERSIONS = ("v5", "v6", "v7")
REVISE_SECONDS = 60.0                  # v7 on the page: longest check-and-repair of the network per Get advice
KNOB_KEYS = ["P", "VA", "VB", "VI", "T"]
KNOB_NAMES = ["Pressure", "Flow of A", "Flow of B", "Inert flow", "Temperature"]
KNOB_UNITS = ["bar", "mL/min", "mL/min", "mL/min", "°C"]
KNOB_DEC = [0, 1, 1, 1, 0]
LOG_COLS = ["pressure_bar", "flow_A_mL_min", "flow_B_mL_min", "flow_I_mL_min", "temperature_C", "catalyst_g"]
SPEC_LABELS = dict(S_C="minimum selectivity to C", dT_ad="maximum adiabatic temperature rise (K)")
REACTIONS = {
    "simple": [dict(name="Main reaction", eq="A + B → C", role="wanted"),
               dict(name="Side reaction", eq="A → D", role="unwanted by-product")],
    "complex": [dict(name="Main reaction", eq="A + B ⇌ C", role="wanted, reversible, gives off heat"),
                dict(name="Over-reaction", eq="C + B → E", role="destroys your product"),
                dict(name="Cracking", eq="A → D", role="unwanted by-product")],
}
PRICE_LABELS = dict(v_C="value of C made (per mmol)", c_feed="cost of A and B fed (per mmol)",
                    c_inert="cost of inert (per mmol)", c_comp="compression (per mmol × ln bar)",
                    c_cat="catalyst (per g per hour)")


def K(c):
    return float(c) + 273.15


def C(k):
    return float(k) - 273.15


def rnd(v, d=3):
    """Round for the page; a value that is not a number (e.g. selectivity of a run that made
    no product) becomes null, which the page shows as a dash."""
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else float(np.round(v, d))


def finite(obj):
    """Recursively turn NaN / inf into null so a response can always be serialised."""
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def user_row(phys5):
    """Internal knob row (K) -> UI units (°C), rounded for display."""
    P, VA, VB, VI, T = (float(v) for v in phys5)
    return dict(P=rnd(P, 1), VA=rnd(VA, 1), VB=rnd(VB, 1), VI=rnd(VI, 1), T=rnd(C(T), 0))


# =================================================================================== the campaign
class Campaign:
    def __init__(self):
        self.lock = threading.Lock()
        self.system = "complex"
        self.wcat, self.a_spec = WCAT, A_SPEC
        self.max_step = MAX_STEP.astype(float).tolist()
        self.lo, self.hi = LOW.astype(float).tolist(), HIGH.astype(float).tolist()   # internal units (K)
        self.fixed = {}                        # knob index -> physical value (K for T)
        self.prices = dict(SYSTEMS[self.system].prices)
        self.prices.pop("disposal", None)
        self.trial_cost, self.hours = None, 8000.0
        self.spec = {}                         # product / safety spec, e.g. {"S_C": 0.9, "dT_ad": 60}
        self.remaining = 10                    # runs the operator can still afford
        self.deactivation = 0.0                # virtual reactor only: activity lost per day, %
        self.version = "v7"                     # advisor: "v7" (default: v6 + the neural network alone, and the reaction
                                               # network is checked and repaired), "v6" (committee correction) or "v5" (GP)
        self.progress = None                   # what a long fit is doing right now (v7 network search), for the page
        self.tradeoff_res = None               # the optional profit vs selectivity curve, while the advice it came from holds
        self.seed = 0
        self.runs = []                         # dicts: P, VA, VB, VI, T (K), W, gc (list), source
        self.vr = None
        self.analysis = None
        self.busy = False
        self.rev = 0                           # bumped on every change of runs or settings
        self.model_lock = threading.RLock()    # one fit at a time

    def _invalidate(self):
        """Call under self.lock whenever the runs or the settings change."""
        self.analysis = None
        self.tradeoff_res = None
        self.rev += 1

    # ------------------------------------------------------------------ helpers
    def S(self):
        S = copy.deepcopy(SYSTEMS[self.system])
        S.prices.update(self.prices)
        return S

    def species(self):
        return SYSTEMS[self.system].species

    def reactor(self, wcat=None):
        wcat = self.wcat if wcat is None else float(wcat)
        key = (self.system, wcat, self.a_spec, self.seed, self.deactivation, tuple(sorted(self.prices.items())))
        if self.vr is None or self.vr_key != key:
            self.vr = VirtualReactor(self.S(), seed=self.seed, wcat=wcat, a_spec=self.a_spec,
                                     deactivation=self.deactivation / 100.0 / 24.0)
            self.vr_key = key
        return self.vr

    def truth_known(self):
        return bool(self.runs) and all(r["source"] == "virtual" for r in self.runs)

    def arrays(self):
        runs = list(self.runs)                 # one snapshot, so both arrays have the same rows
        phys6 = np.array([[r["P"], r["VA"], r["VB"], r["VI"], r["T"], r["W"]] for r in runs], float)
        gc = np.array([r["gc"] for r in runs], float)
        return phys6, gc

    def kpis(self):
        if not self.runs:
            return [], [], [], []
        S = self.S()
        phys6, gc = self.arrays()
        prof = profit_from_gc(S, phys6, gc)
        XA, SC, _ = kpis_from_gc(S, phys6, gc)
        dT = dT_ad_from_gc(S, phys6, gc)
        return prof.tolist(), XA.tolist(), SC.tolist(), dT.tolist()

    def hours_rows(self):
        return np.array([r.get("hours", HOURS_PER_RUN) for r in self.runs], float)

    # ------------------------------------------------------------------ state for the page
    def state(self):
        prof, XA, SC, dT = self.kpis()
        pred = self.analysis["pred"] if self.analysis else [None] * len(self.runs)
        runs = []
        for i, r in enumerate(self.runs):
            runs.append(dict(i=i, **user_row([r["P"], r["VA"], r["VB"], r["VI"], r["T"]]), W=rnd(r["W"], 3),
                             hours=rnd(r.get("hours", HOURS_PER_RUN), 2),
                             gc=[rnd(v, 5) for v in r["gc"]], source=r["source"],
                             profit=rnd(prof[i], 2), XA=rnd(XA[i], 3), SC=rnd(SC[i], 3), dT=rnd(dT[i], 1),
                             pred=rnd(pred[i], 2) if pred[i] is not None else None))
        lo_u, hi_u = list(self.lo), list(self.hi)
        lo_u[4], hi_u[4] = rnd(C(lo_u[4]), 2), rnd(C(hi_u[4]), 2)
        fixed_u = {str(i): (rnd(C(v), 2) if i == 4 else v) for i, v in self.fixed.items()}
        return dict(
            settings=dict(system=self.system, wcat=self.wcat, a_spec=self.a_spec, max_step=self.max_step,
                          lo=lo_u, hi=hi_u, fixed=fixed_u, prices=self.prices, trial_cost=self.trial_cost,
                          hours=self.hours, seed=self.seed, spec=self.spec, remaining=self.remaining,
                          deactivation=self.deactivation, version=self.version),
            meta=dict(knobs=KNOB_KEYS, names=KNOB_NAMES, units=KNOB_UNITS, dec=KNOB_DEC,
                      box_lo=[LOW[0], LOW[1], LOW[2], LOW[3], rnd(C(LOW[4]), 2)], box_hi=[HIGH[0], HIGH[1], HIGH[2], HIGH[3], rnd(C(HIGH[4]), 2)],
                      min_total_flow=MIN_TOTAL_FLOW, species=self.species(), reactions=REACTIONS,
                      descriptions={k: v.description for k, v in SYSTEMS.items()}, price_labels=PRICE_LABELS,
                      spec_labels=SPEC_LABELS, hours_per_run=HOURS_PER_RUN,
                      default_prices={k: {kk: vv for kk, vv in v.prices.items() if kk != "disposal"} for k, v in SYSTEMS.items()}),
            runs=runs, n=len(runs), truth_known=self.truth_known(),
            analysis_ready=self.analysis is not None, busy=self.busy,
            best_idx=(int(np.argmax(prof)) if prof else None),
        )

    # ------------------------------------------------------------------ settings
    def settings(self, b):
        """Every field is validated first; nothing is changed unless everything is valid."""
        with self.lock:
            new = {}
            if "system" in b and b["system"] != self.system:
                if b["system"] not in SYSTEMS:
                    raise ValueError("unknown chemistry")
                if self.runs and len(self.runs[0]["gc"]) != len(SYSTEMS[b["system"]].species):
                    raise ValueError("The runs in the table have GC columns for the other chemistry. "
                                     "Clear the runs before switching chemistry.")
                new["system"] = b["system"]
                new["prices"] = {k: v for k, v in SYSTEMS[b["system"]].prices.items() if k != "disposal"}
            if "wcat" in b:
                w = float(b["wcat"])
                if not 0.01 <= w <= 50:
                    raise ValueError("catalyst mass must be between 0.01 and 50 g")
                new["wcat"] = w
            if "a_spec" in b:
                a = float(b["a_spec"])
                if not 1 <= a <= 2000:
                    raise ValueError("BET area must be between 1 and 2000 m²/g")
                new["a_spec"] = a
            if "max_step" in b:
                ms = [float(v) for v in b["max_step"]]
                if len(ms) != 5 or min(ms) <= 0:
                    raise ValueError("every 'largest change per run' must be a positive number")
                new["max_step"] = ms
            lo, hi = list(self.lo), list(self.hi)
            if "lo" in b and "hi" in b:
                lo, hi = [float(v) for v in b["lo"]], [float(v) for v in b["hi"]]
                if len(lo) != 5 or len(hi) != 5:
                    raise ValueError("safety ranges need a min and a max for each of the 5 knobs")
                lo[4], hi[4] = K(lo[4]), K(hi[4])
                lo = np.clip(lo, LOW, HIGH).tolist()
                hi = np.clip(hi, LOW, HIGH).tolist()
                if any(l >= h for l, h in zip(lo, hi)):
                    raise ValueError("each safety range needs min < max")
                new["lo"], new["hi"] = lo, hi
            if "fixed" in b:
                fx = {}
                for k, v in (b["fixed"] or {}).items():
                    i = int(k)
                    if not 0 <= i < 5:
                        raise ValueError("unknown knob")
                    fx[i] = K(v) if i == 4 else float(v)
                    if not lo[i] - 1e-9 <= fx[i] <= hi[i] + 1e-9:
                        raise ValueError(f"the fixed {KNOB_NAMES[i].lower()} is outside its safety range")
                new["fixed"] = fx
            if "prices" in b:
                prices = dict(new.get("prices", self.prices))
                for k, v in b["prices"].items():
                    if k in PRICE_LABELS:
                        prices[k] = float(v)
                new["prices"] = prices
            if "trial_cost" in b:
                new["trial_cost"] = None if b["trial_cost"] in (None, "") else float(b["trial_cost"])
                if new["trial_cost"] is not None and new["trial_cost"] < 0:
                    raise ValueError("the cost of a run cannot be negative")
            if "hours" in b:
                new["hours"] = float(b["hours"])
                if new["hours"] <= 0:
                    raise ValueError("operating hours per year must be positive")
            if "seed" in b:
                new["seed"] = int(b["seed"])
            if "spec" in b:
                sp = {}
                for k, v in (b["spec"] or {}).items():
                    if k in SPEC_LABELS and v not in (None, ""):
                        sp[k] = float(v)
                        if k == "S_C" and not 0 < sp[k] <= 1:
                            raise ValueError("the minimum selectivity must be between 0 and 1")
                        if k == "dT_ad" and sp[k] <= 0:
                            raise ValueError("the maximum temperature rise must be positive")
                new["spec"] = sp
            if "remaining" in b:
                new["remaining"] = int(b["remaining"])
                if new["remaining"] < 1:
                    raise ValueError("runs you can still afford must be at least 1")
            if "version" in b:
                if b["version"] not in VERSIONS:
                    raise ValueError("the advisor version must be v5, v6 or v7")
                new["version"] = b["version"]
            if "deactivation" in b:
                new["deactivation"] = float(b["deactivation"] or 0.0)
                if not 0 <= new["deactivation"] < 50:
                    raise ValueError("activity loss per day must be between 0 and 50 %")
            for k, v in new.items():
                setattr(self, k, v)
            self._invalidate()
        return self.state()

    # ------------------------------------------------------------------ runs
    def _parse_run(self, b):
        try:
            P, VA, VB, VI, T = (float(b[k]) for k in KNOB_KEYS)
            W = float(b.get("W", self.wcat))
            hours = float(b.get("hours") or HOURS_PER_RUN)
            gc = [float(v) for v in b["gc"]]
        except (KeyError, TypeError, ValueError):
            raise ValueError("every set point and every GC value is needed, as numbers")
        if hours <= 0:
            raise ValueError("the run's duration must be positive")
        if len(gc) != len(self.species()):
            raise ValueError(f"this chemistry has {len(self.species())} GC columns ({', '.join(self.species())})")
        if min(gc) < 0 or sum(gc) <= 0:
            raise ValueError("GC values must be zero or positive")
        gc = np.array(gc)
        if gc.sum() > 1.5:
            gc = gc / 100.0                      # given in %
        gc = gc / gc.sum()
        if gc[-1] <= 0:
            raise ValueError("the inert must show in the GC (it is the internal standard)")
        Tk = K(T)
        row = [P, VA, VB, VI, Tk]
        for i, v in enumerate(row):
            if not LOW[i] - 1e-9 <= v <= HIGH[i] + 1e-9:
                lo, hi = (C(LOW[i]), C(HIGH[i])) if i == 4 else (LOW[i], HIGH[i])
                raise ValueError(f"{KNOB_NAMES[i].lower()} must be between {lo:g} and {hi:g} {KNOB_UNITS[i]}")
        if VA + VB + VI < MIN_TOTAL_FLOW - 1e-9:
            raise ValueError(f"total feed must be at least {MIN_TOTAL_FLOW:g} mL/min for the GC")
        if W <= 0:
            raise ValueError("catalyst mass must be positive")
        return dict(P=P, VA=VA, VB=VB, VI=VI, T=Tk, W=W, hours=hours, gc=gc.tolist(), source=b.get("source", "manual"))

    def add_run(self, b):
        with self.lock:
            self.runs.append(self._parse_run(b))
            self._invalidate()
        return self.state()

    def update_run(self, b):
        with self.lock:
            i = int(b["i"])
            if not 0 <= i < len(self.runs):
                raise ValueError("no such run")
            self.runs[i] = self._parse_run(b)
            self._invalidate()
        return self.state()

    def delete_run(self, b):
        with self.lock:
            i = int(b["i"])
            if not 0 <= i < len(self.runs):
                raise ValueError("no such run")
            self.runs.pop(i)
            self._invalidate()
        return self.state()

    def clear(self, b=None):
        with self.lock:
            self.runs = []
            self._invalidate()
        return self.state()

    def import_csv(self, b):
        text = b.get("text", "")
        rows = [r for r in csv.DictReader(io.StringIO(text)) if any((v or "").strip() for v in r.values())]
        if not rows:
            raise ValueError("no rows found in the file")
        species = self.species()
        need = LOG_COLS + [f"gc_{s}" for s in species]
        miss = [c for c in need if c not in rows[0]]
        if miss:
            raise ValueError("missing columns: " + ", ".join(miss) + ". Expected: " + ", ".join(need))
        new = []
        for r in rows:
            new.append(self._parse_run(dict(P=r["pressure_bar"], VA=r["flow_A_mL_min"], VB=r["flow_B_mL_min"],
                                            VI=r["flow_I_mL_min"], T=r["temperature_C"], W=r["catalyst_g"],
                                            hours=(r.get("hours_h") or "").strip() or HOURS_PER_RUN,
                                            gc=[r[f"gc_{s}"] for s in species], source="manual")))
        with self.lock:
            if b.get("replace", True):
                self.runs = new
            else:
                self.runs.extend(new)
            self._invalidate()
        return self.state()

    def example(self, b):
        """The 5 screening runs an operator does first, measured on the virtual reactor."""
        with self.lock:
            if "seed" in b:
                self.seed = int(b["seed"])
            vr = self.reactor()
            rows = screening_runs(base_case(self.seed), self.max_step)
            self.runs = []
            for row in rows:
                r = vr.run(scale(row)[0])
                self.runs.append(dict(P=row[0], VA=row[1], VB=row[2], VI=row[3], T=row[4], W=self.wcat,
                                      hours=r["hours"], gc=r["gc"].tolist(), source="virtual"))
            self._invalidate()
        return self.state()

    # ------------------------------------------------------------------ the analysis
    def _advisor(self):
        """The advisor is kept between calls while the settings are unchanged, so a refit
        after one more run starts from the previous kinetics (1-3 s instead of 10-15 s)."""
        S = self.S()
        phys6, gc = self.arrays()
        X = scale(phys6[:, :5])
        prof = profit_from_gc(S, phys6, gc)
        lo = (np.array(self.lo) - LOW) / (HIGH - LOW)
        hi = (np.array(self.hi) - LOW) / (HIGH - LOW)
        fixed = {i: (v - LOW[i]) / (HIGH[i] - LOW[i]) for i, v in self.fixed.items()}
        key = (self.system, self.a_spec, self.seed, float(phys6[-1, 5]), tuple(self.max_step), tuple(self.lo),
               tuple(self.hi), tuple(sorted(fixed.items())), tuple(sorted(self.prices.items())),
               tuple(sorted(self.spec.items())), self.version)
        adv = getattr(self, "_adv", None)
        if adv is None or getattr(self, "_adv_key", None) != key:
            v7 = self.version == "v7"
            adv = SmartAdvisor(S, wcat=phys6[-1, 5], a_spec=self.a_spec, seed=self.seed, lo=lo, hi=hi,
                               fixed=fixed, max_step=self.max_step, spec=self.spec, remaining=self.remaining,
                               correction={"v5": "gp", "v6": "committee", "v7": "hybrid"}[self.version],
                               revise=v7, revise_seconds=REVISE_SECONDS if v7 else None)
            self._adv, self._adv_key = adv, key
        adv.remaining = self.remaining
        t0 = time.time()

        def progress(info):
            self.progress = dict(message=str(info.get("message", "")), elapsed=round(time.time() - t0, 1))

        self.progress = None
        try:
            if getattr(adv, "revise", False):
                adv.update(X, gc, prof, wcat_rows=phys6[:, 5], hours=self.hours_rows(), progress=progress)
            else:
                adv.update(X, gc, prof, wcat_rows=phys6[:, 5], hours=self.hours_rows())
        finally:
            self.progress = None
        return S, adv, X, prof

    def analyse(self, b=None):
        if not self.model_lock.acquire(blocking=False):
            raise ValueError("The advisor is still working on the previous request. Wait for it to finish.")
        try:
            return self._analyse(b)
        finally:
            self.model_lock.release()

    def _analyse(self, b):
        with self.lock:
            if len(self.runs) < 3:
                raise ValueError("at least 3 runs are needed before the model can say anything useful")
            self.busy = True
            rev = self.rev
        try:
            S, adv, X, prof = self._advisor()
            rng = np.random.default_rng(self.seed)
            x_next, info = adv.suggest(X[-1], rng)
            ib = adv.recommend()
            mu_data, sd_data = adv.predict(X)
            x_last, x_best = X[-1], X[ib]
            an = dict(
                pred=mu_data.tolist(),
                best_idx=ib,
                next=self._next_block(adv, S, X, info, x_next, x_last, x_best, mu_data),
                plan=self._plan_block(info),
                worth=self._worth_block(info, mu_data[ib]),
                today=user_row(unscale(x_best)[0]),
                chemistry=self._chemistry_block(adv, S, X),
                arrhenius=self._arrhenius_block(adv, S),
                parity=self._parity_block(adv, S),
                landscape=self._landscape_block(adv, x_next, x_last, x_best, info["model_opt"], b.get("ref", "next") if b else "next"),
                series=dict(measured=prof.tolist(), best=np.maximum.accumulate(prof).tolist(),
                            model=mu_data.tolist(), model_sd=sd_data.tolist()),
                x_next=x_next.tolist(), x_opt=np.asarray(info["model_opt"]).tolist(), seconds=None,
            )
            with self.lock:
                if self.rev != rev:            # the table or the settings changed meanwhile
                    raise ValueError("The runs or settings changed while the advice was being computed. "
                                     "Press Get advice again.")
                self.analysis = an
        finally:
            with self.lock:
                self.busy = False
        return dict(analysis=an, state=self.state())

    def _moves(self, x_from, x_to):
        a, b_ = unscale(x_from)[0], unscale(x_to)[0]
        out = []
        for i in range(5):
            fa, fb = (C(a[i]), C(b_[i])) if i == 4 else (a[i], b_[i])
            d = fb - fa
            out.append(dict(knob=KNOB_KEYS[i], name=KNOB_NAMES[i], unit=KNOB_UNITS[i], frm=rnd(fa, KNOB_DEC[i]),
                            to=rnd(fb, KNOB_DEC[i]), delta=rnd(d, KNOB_DEC[i]),
                            changed=bool(abs(d) >= 0.02 * (HIGH[i] - LOW[i]))))
        return out

    def _next_block(self, adv, S, X, info, x_next, x_last, x_best, mu_data):
        gain = info["pred"] - mu_data[int(np.argmax(mu_data))]
        _, _, parts = adv.predict(x_next[None], parts=True)
        if info.get("mode") == "recover":
            why = ("None of the runs within one step is likely to meet your spec, so this is the step most "
                   "likely to get back within spec while heading toward the predicted best.")
            kind = "recover"
        elif info.get("mode") == "stay":
            why = ("Keep running these settings: nothing within one step is expected to earn more over the runs "
                   "you have left, so the best use of the next run is production at the best point.")
            kind = "stay"
        elif info.get("mode") == "walk":
            why = ("Nothing within one step beats your best run yet, but over the runs you have left the model "
                   "expects more from heading toward the predicted best: this is the step that way.")
            kind = "walk"
        elif gain > 1.64 * info["sd"]:
            why = f"The chemistry model is fairly sure this beats your best run, by about {gain:.1f} per hour."
            kind = "improve"
        elif info["toward_opt"]:
            why = "It is the biggest allowed step toward where the model expects the best profit."
            kind = "walk"
        elif parts["sd_phys"][0] > parts["sd_corr"][0]:
            why = ("The versions of your chemistry that fit your data disagree most about this region, and it "
                   "could beat your best run. This run settles it.")
            kind = "check"
        else:
            why = ("Your data and the textbook physics disagree a little around here. This run checks whether "
                   "the extra profit the model sees is real.")
            kind = "check"
        return dict(**user_row(unscale(x_next)[0]), pred=rnd(info["pred"], 2),
                    lo=rnd(info["pred"] - 1.64 * info["sd"], 2), hi=rnd(info["pred"] + 1.64 * info["sd"], 2),
                    ei=rnd(info["ei"], 3), why=why, kind=kind, moves=self._moves(x_last, x_next),
                    same_as_last=bool(np.max(np.abs(x_next - x_last)) < 1e-6),
                    p_spec=rnd(info.get("p_spec", 1.0), 3), spec=dict(adv.spec),
                    traj_profit=[rnd(v, 2) for v in info.get("traj_profit", [])])

    def _plan_block(self, info):
        return dict(**user_row(unscale(info["model_opt"])[0]), pred=rnd(info["model_opt_pred"], 2),
                    lo=rnd(info["model_opt_pred"] - 1.64 * info["model_opt_sd"], 2),
                    hi=rnd(info["model_opt_pred"] + 1.64 * info["model_opt_sd"], 2),
                    steps_away=int(info["steps_away"]))

    def _worth_block(self, info, best_pred):
        far = max(0.0, info["model_opt_pred"] - 1.64 * info["model_opt_sd"] - best_pred)
        out = dict(ei=rnd(info["ei"], 2), far=rnd(far, 2), hours=self.hours, trial_cost=self.trial_cost)
        if self.trial_cost is not None:
            n = max(int(info["steps_away"]), 1)
            yearly, cost = far * self.hours, self.trial_cost * n
            out.update(runs=n, yearly=rnd(yearly, 0), cost=rnd(cost, 0), verdict="yes" if yearly > cost else "no",
                       text=(f"Over {self.hours:,.0f} operating hours a year the remaining gain is worth about "
                             f"{yearly:,.0f}, against {n} run(s) at {self.trial_cost:,.0f} each = {cost:,.0f}."))
        elif far < 0.02 * max(abs(best_pred), 1):
            out.update(verdict="no", text="The remaining gain is under 2 % of your current profit: you are close "
                                          "to the best these knobs allow.")
        else:
            out.update(verdict="yes", text="Worth running. Enter a cost per run and your operating hours to weigh it in money.")
        return out

    @staticmethod
    def _rx_list(adv, S):
        """The reactions the advisor is working with, as cards for the page, and whether they are
        still the network it was given (v7 may have changed it; then there is no 'true' line to show)."""
        net, start = getattr(adv, "S", S), getattr(adv, "S_start", None)
        if start is None or net is start:
            return REACTIONS[S.name], False
        given = {r.key(): rx for r, rx in zip(D.reactions_from_system(start), REACTIONS[S.name])}
        out = []
        for j, r in enumerate(D.reactions_from_system(net)):
            eq = r.equation().replace("<=>", "⇌").replace("->", "→")
            rx = given.get(r.key())
            out.append(dict(name=rx["name"] if rx else f"Reaction {j + 1}", eq=eq,
                            role=rx["role"] if rx else "found from your GC data (not in the network you started with)"))
        return out, True

    def _chemistry_block(self, adv, S, X):
        th = adv.kin.split(adv.kin.samples(400, np.random.default_rng(1)))["kin"]
        Ea = th[:, 1::2] * R * T_REF / 1000.0
        P, VA, VB, VI, T = unscale(X[-1])[0]
        reactions = []
        rxs, changed = self._rx_list(adv, S)
        truth = self.truth_known() and not changed
        k_t, E_t = theta_to_physical(S.theta_true)
        for j, rx in enumerate(rxs):
            e_med, e_lo, e_hi = np.median(Ea[:, j]), *np.percentile(Ea[:, j], [10, 90])
            f10 = np.exp(e_med * 1000 / R * (1 / T - 1 / (T + 10)))
            reactions.append(dict(**rx, Ea=rnd(e_med, 0), Ea_lo=rnd(e_lo, 0), Ea_hi=rnd(e_hi, 0),
                                  factor10=rnd(f10, 2), Ea_true=(rnd(E_t[j], 0) if truth else None)))
        notes = []
        e = np.median(Ea, 0)
        worse = [] if changed else [rxs[j]["name"].lower() for j in range(1, len(rxs)) if e[j] > e[0]]
        if worse:
            notes.append(f"Heat helps the unwanted {' and '.join(worse)} more than the main reaction: hotter is "
                         f"faster, but you lose product. There is a best temperature.")
        if S.thermo[0] is not None:
            dH, dS = S.thermo[0]
            r = VI / max(VA, 1e-9)
            nA, nB = 1.0, VB / VA
            x = 0.9
            tot = nA + nB - x + r
            K_need = x * tot / (P * (nA - x) * max(nB - x, 1e-3))
            T90 = -dH / (R * (np.log(K_need) - dS / R))
            notes.append(f"The main reaction is reversible and gives off heat: at {P:.0f} bar, above about "
                         f"{C(T90):.0f} °C the equilibrium itself stops more than 90 % of A turning into C, however "
                         f"much catalyst you use.")
        notes.append("Diluting with inert lowers every partial pressure and slows all reactions; it only pays "
                     "when it improves selectivity.")
        d = adv.kin.describe(T=T)
        if d["model"] == "plain":
            notes.append("Your GC readings are explained by the rate laws alone: no sign yet of the product "
                         "slowing the reactions or of the catalyst losing activity.")
        else:
            notes.append(f"Your data show the product slowing the reactions (product inhibition): with "
                         f"{d['K_C']:.3g} /bar of product partial pressure the rates halve at about "
                         f"{1 / max(d['K_C'], 1e-9):.1f} bar of C (likely {d['K_C_lo']:.2g}-{d['K_C_hi']:.2g} /bar). "
                         f"This is why pushing conversion to the limit pays less than the rate laws alone suggest.")
        if "loss_per_day" in d:
            if d["loss_per_day"] > 0:
                notes.append(f"The catalyst is losing activity: about {100 * d['loss_per_day']:.1f} % per day on stream "
                             f"(likely {100 * d['loss_lo']:.1f}-{100 * d['loss_hi']:.1f} %). The advice already allows for it; "
                             f"plan a regeneration or a fresh bed when the loss becomes too costly.")
            else:
                notes.append(f"The catalyst is still breaking in: activity rising about {-100 * d['loss_per_day']:.1f} % per day on stream.")
        elif not adv.kin.has_replicates and len(adv.X) >= 8:
            notes.append("Whether the catalyst is losing activity cannot be told yet: no set point has been repeated. "
                         "A check run (repeat an earlier run) would show it.")
        if getattr(adv, "committee", None) is not None:
            w = adv.committee.weights
            names = {"none": "no correction (the physics alone)", "gp": "the Gaussian process",
                     "nn": "the neural-network ensemble",
                     "direct": "the neural network alone (no physics: profit straight from the knobs)"}
            top = max(w, key=w.get)
            parts = ", ".join(f"{names.get(k, k).split(' (')[0]} {100 * v:.0f} %" for k, v in w.items())
            notes.append(f"Correction to the physics ({self.version} committee): {parts}. The model that best predicts runs "
                         f"it was not trained on gets the most say; here that is {names.get(top, top)}.")
        resid = (adv.kin.predict_gc(physical(adv.X, adv.W), t=adv.t_rows) - adv.gc)[:, :-1]
        Tr = unscale(adv.X)[:, 4]
        return dict(reactions=reactions, notes=notes, miss_pct=rnd(100 * np.median(np.abs(resid)), 2),
                    T_range=[rnd(C(Tr.min()), 0), rnd(C(Tr.max()), 0)], at_T=rnd(C(T), 0),
                    truth_known=truth, model=d["model"],
                    K_C=rnd(d.get("K_C"), 4), loss_per_day=rnd(d.get("loss_per_day"), 4),
                    has_replicates=bool(adv.kin.has_replicates), network=self._network_block(adv, changed))

    @staticmethod
    def _network_block(adv, changed):
        """v7: the reaction network the advisor works with, what it changed and why (None for v5 / v6)."""
        if not getattr(adv, "revise", False):
            return None
        m = getattr(adv, "misfit", None) or {}
        pretty = lambda lines: [ln.replace("<=>", "⇌").replace("->", "→") for ln in lines]
        return dict(lines=pretty(D.format_network(D.reactions_from_system(adv.S))),
                    given=pretty(D.format_network(D.reactions_from_system(adv.S_start))),
                    changed=bool(changed),
                    revisions=[dict(run=r.get("run"), change=r.get("change"), words=r.get("words")) for r in adv.revisions],
                    log=[dict(run=r.get("run"), outcome=r.get("outcome"), words=r.get("words")) for r in adv.revision_log[-6:]],
                    check=m.get("words"), miss_x_noise=rnd(m.get("x_noise"), 2))

    def _arrhenius_block(self, adv, S):
        Tg = np.linspace(450, 850, 60)
        draws = adv.kin.split(adv.kin.samples(200, np.random.default_rng(0)))["kin"]
        out = dict(x=(1000 / Tg).round(4).tolist(), T_C=[rnd(C(t), 0) for t in Tg], reactions=[])
        rxs, changed = self._rx_list(adv, S)
        for j, rx in enumerate(rxs):
            lg = lambda th: (th[2 * j] - th[2 * j + 1] * (T_REF / Tg - 1)) / np.log(10)
            band = np.array([lg(d) for d in draws])
            lo, hi = np.percentile(band, [5, 95], 0)
            out["reactions"].append(dict(name=rx["name"], eq=rx["eq"], learned=lg(adv.kin.theta_kin).round(3).tolist(),
                                         lo=lo.round(3).tolist(), hi=hi.round(3).tolist(),
                                         truth=(lg(S.theta_true).round(3).tolist() if self.truth_known() and not changed else None)))
        return out

    def _parity_block(self, adv, S):
        pred = adv.kin.predict_gc(physical(adv.X, adv.W), t=adv.t_rows)
        return dict(species=S.species[:-1], meas=adv.gc[:, :-1].round(5).tolist(), pred=pred[:, :-1].round(5).tolist())

    def landscape(self, b):
        """Only the what-if charts, around a different reference (reuses the fitted advisor)."""
        if not self.model_lock.acquire(blocking=False):
            raise ValueError("The advisor is still working on the previous request. Wait for it to finish.")
        try:
            with self.lock:
                an = self.analysis
                if an is None or getattr(self, "_adv", None) is None:
                    raise ValueError("get advice first")
                adv = self._adv
                X = scale(self.arrays()[0][:, :5])
            ls = self._landscape_block(adv, np.array(an["x_next"]), X[-1], X[an["best_idx"]], np.array(an["x_opt"]),
                                       b.get("ref", "next"))
            with self.lock:
                if self.analysis is an:
                    an["landscape"] = ls
            return dict(landscape=ls)
        finally:
            self.model_lock.release()

    def _landscape_block(self, adv, x_next, x_last, x_best, x_opt, ref):
        x_ref = {"next": x_next, "best": x_best, "plan": x_opt}.get(ref, x_next)
        blo, bhi = step_box(x_last, adv.step, adv.lo, adv.hi, adv.fixed)
        out = dict(ref=ref, knobs=[])
        for i in range(5):
            g = np.linspace(adv.lo[i], adv.hi[i], 41)
            Xg = np.tile(x_ref, (len(g), 1))
            Xg[:, i] = g
            ok = feasible(unscale(Xg))
            mu, sd = adv.predict(Xg)
            mu[~ok] = np.nan
            mu[mu < -900] = np.nan                 # the physics gave no answer there
            xs = unscale(Xg)[:, i]
            if i == 4:
                xs = xs - 273.15
            cur = unscale(x_ref)[0][i]
            reach = [blo[i], bhi[i]]
            reach = [LOW[i] + v * (HIGH[i] - LOW[i]) for v in reach]
            if i == 4:
                cur, reach = C(cur), [C(reach[0]), C(reach[1])]
            out["knobs"].append(dict(knob=KNOB_KEYS[i], name=KNOB_NAMES[i], unit=KNOB_UNITS[i],
                                     x=xs.round(2).tolist(), mu=[None if np.isnan(v) else rnd(v, 2) for v in mu],
                                     sd=[rnd(v, 2) for v in sd], cur=rnd(cur, 2), reach=[rnd(reach[0], 2), rnd(reach[1], 2)],
                                     fixed=i in adv.fixed))
        return out

    # ------------------------------------------------------------------ optional: profit vs selectivity
    def tradeoff(self, b=None):
        """The profit vs selectivity trade-off curve of the advisor as last fitted (smart/pareto.py).
        Optional: nothing else on the page uses it, and it changes nothing in the advice."""
        if not self.model_lock.acquire(blocking=False):
            raise ValueError("The advisor is still working on the previous request. Wait for it to finish.")
        try:
            with self.lock:
                if self.analysis is None or getattr(self, "_adv", None) is None:
                    raise ValueError("Get advice first: the curve is drawn from the model the advice was based on.")
                an, rev, adv = self.analysis, self.rev, self._adv
            res = finite(pareto_tradeoff(adv, n_points=12, seed=self.seed))
            for pt in res["points"]:
                pt["T_C"] = rnd(pt.get("T_C"), 0)
            with self.lock:
                if self.rev != rev or self.analysis is not an:
                    raise ValueError("The runs or settings changed meanwhile. Get advice again, then ask for the curve.")
                self.tradeoff_res = res
            return dict(tradeoff=res)
        finally:
            self.model_lock.release()

    def tradeoff_aim(self, b):
        """'Aim for this point': set the minimum selectivity that makes the advisor look for the best
        profit at that purity (the other specs are kept).  The advice must be asked for again."""
        with self.lock:
            res = self.tradeoff_res
            if res is None:
                raise ValueError("Show the trade-off curve first.")
            i = int(b.get("i", -1))
            if not 0 <= i < len(res["points"]):
                raise ValueError("no such point on the curve")
            spec = spec_for_point(res["points"][i], spec=self.spec)
        spec["S_C"] = float(np.clip(round(spec["S_C"], 4), 1e-4, 1.0))
        return self.settings(dict(spec=spec))

    # ------------------------------------------------------------------ the virtual reactor
    def simulate(self, b):
        """Run the suggested next run on the virtual reactor and add it to the table."""
        if self.analysis is None:
            self.analyse()
        with self.lock:
            x = np.array(self.analysis["x_next"])
            W = self.runs[-1]["W"]             # the bed that is loaded, as the advice assumed
            vr = self.reactor(W)
            r = vr.run(x, hours=self.runs[-1].get("hours", HOURS_PER_RUN))
            P, VA, VB, VI, T = unscale(x)[0]
            self.runs.append(dict(P=P, VA=VA, VB=VB, VI=VI, T=T, W=W, hours=r["hours"], gc=r["gc"].tolist(),
                                  source="virtual"))
            self._invalidate()
        return self.state()

    # ------------------------------------------------------------------ exports
    def export_log(self):
        species = self.species()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["run"] + LOG_COLS + [f"gc_{s}" for s in species] + ["hours_h"])
        for i, r in enumerate(self.runs, 1):
            w.writerow([i, r["P"], r["VA"], r["VB"], r["VI"], round(C(r["T"]), 2), r["W"]] + [round(v, 5) for v in r["gc"]]
                       + [round(r.get("hours", HOURS_PER_RUN), 2)])
        return buf.getvalue()

    def export_next(self):
        if self.analysis is None:
            raise ValueError("get advice first")
        nx = self.analysis["next"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(LOG_COLS + ["expected_profit_per_h", "low", "high"])
        w.writerow([nx["P"], nx["VA"], nx["VB"], nx["VI"], nx["T"], self.runs[-1]["W"], nx["pred"], nx["lo"], nx["hi"]])
        return buf.getvalue()

# =================================================================================== the reaction finder
FINDER_EXAMPLES = [
    dict(file="mystery_noisy.csv", answer="mystery_reactions.txt", mode="discover",
         title="Mystery reactions, with GC noise",
         text="30 runs of a kinetic study, calculated from three reactions that are not the built-in ones, with the "
              "GC's noise added. Press 'Find the reactions' (about 1-2 minutes) and compare with the answer."),
    dict(file="mystery_exact.csv", answer="mystery_reactions.txt", mode="discover",
         title="Mystery reactions, exact",
         text="The same 30 runs calculated exactly from the reactions (no noise)."),
    dict(file="complex_noisy.csv", answer="candidates_truth.txt", mode="rank",
         title="Complex network, with GC noise",
         text="30 runs calculated from A + B ⇌ C, C + B → E, A → D with GC noise. Rank the four example candidates "
              "(the true network, one quite close, one relatively the same, one very different)."),
    dict(file="complex_exact.csv", answer="candidates_truth.txt", mode="rank",
         title="Complex network, exact",
         text="The same 30 runs calculated exactly (no noise)."),
    dict(file="complex_real.csv", answer="candidates_truth.txt", mode="rank",
         title="Complex network on the virtual 'real' reactor",
         text="The same 30 runs measured on the virtual reactor: product and reactant block catalyst sites (physics "
              "the plain rate laws do not have), plus GC noise."),
    dict(file="complex_short_campaign.csv", answer="candidates_truth.txt", mode="rank",
         title="Start of a campaign: only 8 runs",
         text="The 5 screening runs plus 3 step-limited runs on the virtual reactor. Ranking the candidates takes "
              "about 10 s; the automatic search may need several new starts here (minutes)."),
    dict(file="zero_order_noisy.csv", answer="candidates_close.txt", mode="discover",
         title="A rate law that does not follow its equation",
         text="30 runs where the over-reaction C + B → E runs at r = k P_C (zero order in B), with GC noise. "
              "Can the automatic search find the order? (2-3 minutes)"),
]
FINDER_CANDIDATES = [
    dict(key="truth", file="candidates_truth.txt", name="True network"),
    dict(key="close", file="candidates_close.txt", name="Quite close"),
    dict(key="similar", file="candidates_similar.txt", name="Relatively the same"),
    dict(key="far", file="candidates_far.txt", name="Very different"),
]
FINDER_MAX_SECONDS = 900.0              # longest automatic search the page waits for


def _finder_child():
    """The reaction finder in a process of its own (started by Finder._run as
    'python -c "import reactor_ui; reactor_ui._finder_child()"'): the job arrives pickled on stdin;
    progress lines and the result go back as one JSON object per line on stdout.  A plain child
    process rather than multiprocessing, which on Windows would re-run the script that started
    the server (or a test) inside the child."""
    import pickle
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    out = sys.stdout
    sys.stdout = sys.stderr                     # nothing but the protocol on the real stdout

    def send(obj):
        out.write(json.dumps(finite(obj)) + "\n")
        out.flush()

    try:
        mode, data, cands, start, max_seconds = pickle.load(sys.stdin.buffer)
        progress = lambda info: send(dict(log=[float(info.get("elapsed_s", 0.0)), str(info.get("message", ""))]))
        if mode == "rank":
            res = D.rank_candidates([(name, D.parse_network(text)) for name, text in cands], data, progress=progress)
        else:
            res = D.discover(data, start=start, progress=progress, max_seconds=max_seconds)
        send(dict(done=res))
    except ValueError as e:
        send(dict(error=str(e)))
    except Exception as e:
        send(dict(error=f"{type(e).__name__}: {e}"))


def _pretty(lines):
    return [ln.replace("<=>", "⇌").replace("->", "→") for ln in lines]


class Finder:
    """Runs the reaction finder (smart/discovery.py) in the background for the page: one job at a
    time, with its progress messages, so a search of a minute or two does not block the page."""

    def __init__(self, campaign):
        self.camp = campaign
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()       # one next-run search at a time
        self.job = None
        self.n = 0

    # ------------------------------------------------------------------ inputs
    def examples(self, b=None):
        ex = []
        for e in FINDER_EXAMPLES:
            fp = os.path.join(EX_DIR, e["file"])
            if os.path.isfile(fp):
                ex.append(dict(file=e["file"], title=e["title"], text=e["text"], mode=e["mode"]))
        cands = []
        for c in FINDER_CANDIDATES:
            fp = os.path.join(EX_DIR, c["file"])
            if os.path.isfile(fp):
                with open(fp, encoding="utf-8") as fh:
                    cands.append(dict(key=c["key"], name=c["name"], text=fh.read().strip()))
        return dict(examples=ex, candidates=cands, max_seconds=FINDER_MAX_SECONDS)

    def example_file(self, name):
        if name not in {e["file"] for e in FINDER_EXAMPLES} | {c["file"] for c in FINDER_CANDIDATES} | {"mystery_reactions.txt"}:
            raise ValueError("no such example file")
        with open(os.path.join(EX_DIR, name), encoding="utf-8") as fh:
            return fh.read()

    def _data(self, b):
        src = b.get("source", "example")
        if src == "example":
            e = next((e for e in FINDER_EXAMPLES if e["file"] == b.get("example")), None)
            if e is None:
                raise ValueError("choose one of the example files")
            return D.load_data(os.path.join(EX_DIR, e["file"])), e["title"], e
        if src == "upload":
            text = b.get("text") or ""
            if not text.strip():
                raise ValueError("the uploaded file is empty")
            return D.load_data(text), b.get("name") or "your file", None
        if src == "runs":
            with self.camp.lock:
                n = len(self.camp.runs)
                if n < 3:
                    raise ValueError("there are fewer than 3 runs in the table: load or type some runs first")
                text = self.camp.export_log()
                a_spec = self.camp.a_spec
            return D.load_data(text, a_spec=a_spec), f"the {n} runs in the table", None
        raise ValueError("unknown data source")

    @staticmethod
    def _network(text, name):
        try:
            net = D.parse_network(text)
        except ValueError as e:                # ReactionError is a ValueError: say which candidate
            raise ValueError(f"'{name}': {e}") from None
        if not net:
            raise ValueError(f"'{name}' has no reactions")
        return net

    # ------------------------------------------------------------------ the job
    def start(self, b):
        mode = b.get("mode")
        if mode not in ("discover", "rank"):
            raise ValueError("choose 'find the reactions' or 'rank candidates'")
        data, label, ex = self._data(b)
        cands, start = [], None
        if mode == "rank":
            seen = set()
            for i, c in enumerate(b.get("candidates") or [], 1):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                name = (c.get("name") or "").strip() or f"candidate {i}"
                while name in seen:
                    name += " (2)"
                seen.add(name)
                self._network(text, name)              # a typing error shows now, not in the background
                cands.append((name, text))
            if not cands:
                raise ValueError("type at least one candidate network (one reaction per line, e.g. A + B <=> C)")
        elif (b.get("start") or "").strip():
            self._network(b["start"], "the network to repair")
            start = b["start"]
        with self.lock:
            if self.job is not None and self.job["status"] == "running":
                raise ValueError("The reaction finder is still working on the previous request: wait, or press Stop.")
            self.n += 1
            job = dict(id=self.n, mode=mode, status="running", log=[], started=time.time(), finished=None,
                       label=label, n_runs=int(len(data["gc"])), result=None, error=None, cancel=False,
                       repair=start is not None, example=ex["file"] if ex else None)
            self.job = job
        threading.Thread(target=self._run, args=(job, data, cands, start, ex), daemon=True).start()
        return self.status()

    def _run(self, job, data, cands, start, ex):
        """Runs the search in a separate process (_finder_child), so that it neither slows the
        advice (Python runs one thread at a time) nor has to be waited for when Stop is pressed."""
        import pickle
        import subprocess

        def finish(**kw):
            with self.lock:
                job.update(finished=time.time(), **kw)

        proc = None
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            proc = subprocess.Popen([sys.executable, "-c", f"import sys; sys.path.insert(0, {here!r}); "
                                     "import reactor_ui; reactor_ui._finder_child()"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=here)
            with self.lock:
                job["proc"] = proc
                cancelled = job["cancel"]
            if cancelled:
                proc.kill()
            proc.stdin.write(pickle.dumps((job["mode"], data, cands, start, FINDER_MAX_SECONDS)))
            proc.stdin.close()
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "log" in msg:
                    with self.lock:
                        job["log"].append(dict(t=round(msg["log"][0], 1), msg=msg["log"][1]))
                        del job["log"][:-300]
                elif "done" in msg:
                    res = msg["done"]
                    return finish(status="done", result=self._view(res, data, job["mode"], ex), raw=res, data=data)
                elif "error" in msg:
                    return finish(status="error", error=msg["error"])
            if job["cancel"]:
                return finish(status="stopped", error="Stopped.")
            finish(status="error", error="the reaction finder stopped unexpectedly")
        except Exception as e:                  # keep the page alive whatever happens
            if job["cancel"]:
                return finish(status="stopped", error="Stopped.")
            finish(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()

    def stop(self, b=None):
        with self.lock:
            j = self.job
            if j is not None and j["status"] == "running":
                j["cancel"] = True
                if j.get("proc") is not None:
                    j["proc"].kill()          # the reading thread then sees the end of the output
        return self.status()

    def status(self, b=None):
        with self.lock:
            j = self.job
            if j is None:
                return dict(job=None)
            end = j["finished"] or time.time()
            pub = {k: v for k, v in j.items() if k not in ("raw", "data", "log", "cancel", "proc")}
            pub["log"] = list(j["log"][-12:])
            pub["elapsed"] = round(end - j["started"], 1)
            pub["stopping"] = bool(j["cancel"]) and j["status"] == "running"
            return dict(job=pub)

    # ------------------------------------------------------------------ what the page shows
    @staticmethod
    def _fit_view(f):
        keep = ("name", "rank", "equations", "reactions", "verdict", "x_noise", "bic", "delta_bic", "weight",
                "tied_with_best", "fits_noise", "model", "model_words", "n_params", "n_reactions", "unexplained",
                "not_made", "differs_by", "inhibition")
        out = {k: f.get(k) for k in keep if k in f}
        out["equations"] = _pretty(f.get("equations", []))
        out["miss"] = {sp: dict(x_noise=rnd(m["x_noise"], 2), mol_pct=rnd(m["mol_pct"], 3))
                       for sp, m in (f.get("miss") or {}).items()}
        out["rate_constants"] = [dict(reaction=r["reaction"].replace("<=>", "⇌").replace("->", "→"),
                                      rate_law=r["rate_law"], k=r["k_ref"], k_lo=r["k_ref_low"], k_hi=r["k_ref_high"],
                                      units=r["k_units"], Ea=rnd(r["Ea_kJ"], 1), Ea_sd=rnd(r["Ea_sd_kJ"], 1))
                                 for r in f.get("rate_constants", [])]
        return finite(out)

    def _view(self, res, data, mode, ex):
        fits = res["ranking"] if mode == "rank" else [res["best"]] + list(res.get("runners_up", []))
        best = fits[0]
        sp = list(data["species"][:-1])
        pred = np.asarray(best.get("predicted_gc") or np.zeros_like(data["gc"]))
        out = dict(mode=mode, networks=[self._fit_view(f) for f in fits[:8]], best=best["name"],
                   decided=bool(res.get("decided", True)), undecided=list(res.get("undecided", [])),
                   summary=list(res.get("summary", [])), cautions=list(res.get("cautions", [])),
                   notes=list(res.get("notes", [])), detected={k: rnd(v, 3) for k, v in (res.get("detected") or {}).items()},
                   n_runs=int(res.get("n_runs", len(data["gc"]))), n_settings=res.get("n_settings"),
                   seconds=rnd(res.get("seconds"), 1), n_fits=res.get("n_fits"), restarts=res.get("restarts"),
                   path=list(res.get("path", [])) if mode == "discover" else [],
                   stopped_early=bool(res.get("stopped_early", False)), complete=res.get("complete"),
                   parity=dict(species=sp, meas=np.asarray(data["gc"])[:, :len(sp)].round(5).tolist(),
                               pred=pred[:, :len(sp)].round(5).tolist()),
                   answer=None)
        if ex and ex.get("answer"):
            ans = D.read_network_file(os.path.join(EX_DIR, ex["answer"]))
            found = D.parse_network(best["reactions"]) if best.get("reactions") else []
            out["answer"] = dict(file=ex["answer"], equations=_pretty(D.format_network(ans)),
                                 match=bool(found) and D.network_key(found) == D.network_key(ans))
        return finite(out)

    def next_run(self, b):
        """The next run that would best tell the two best networks apart (discovery.discriminating_run)."""
        with self.lock:
            j = self.job
            if j is None or j["status"] != "done":
                raise ValueError("run the reaction finder first")
            raw, data = j["raw"], j["data"]
        n_nets = len(raw.get("ranking") or [raw.get("best")] + list(raw.get("runners_up", [])))
        if n_nets < 2:
            raise ValueError("there is only one network: give at least two candidates to compare")
        if not self.run_lock.acquire(blocking=False):
            raise ValueError("still working on the previous next-run search")
        try:
            step = bool(b.get("from_last"))
            nr = D.discriminating_run(raw, data, x_last=data["phys"][-1, :5] if step else None,
                                      max_step=np.array(self.camp.max_step, float) if step else None)
        finally:
            self.run_lock.release()
        keep = ("run", "networks", "predicted", "difference_x_noise", "separation", "separation_uncertain",
                "expected_bic_gap", "best_so_far", "step_limited", "message")
        return dict(next_run=finite({k: nr.get(k) for k in keep}))



CAMP = Campaign()
FINDER = Finder(CAMP)
ROUTES = {
    "/api/settings": CAMP.settings, "/api/runs/add": CAMP.add_run, "/api/runs/update": CAMP.update_run,
    "/api/runs/delete": CAMP.delete_run, "/api/runs/clear": CAMP.clear, "/api/runs/import": CAMP.import_csv,
    "/api/example": CAMP.example, "/api/analyse": CAMP.analyse, "/api/simulate": CAMP.simulate,
    "/api/landscape": CAMP.landscape, "/api/tradeoff": CAMP.tradeoff, "/api/tradeoff/aim": CAMP.tradeoff_aim,
    "/api/finder/start": FINDER.start, "/api/finder/stop": FINDER.stop, "/api/finder/next_run": FINDER.next_run,
}
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".svg": "image/svg+xml", ".png": "image/png"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype, status=200, download=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        try:
            body = json.dumps(obj, allow_nan=False)
        except ValueError:                          # a stray NaN somewhere must never take the page down
            body = json.dumps(finite(obj))
        self._send(body, "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            path = "/index.html"
        try:
            if path == "/api/state":
                return self._json(CAMP.state())
            if path == "/api/analysis":
                if CAMP.analysis is None:
                    raise ValueError("no advice yet")
                return self._json(dict(analysis=CAMP.analysis))
            if path == "/api/export/log.csv":
                return self._send(CAMP.export_log(), "text/csv", download="reactor_log.csv")
            if path == "/api/export/next_run.csv":
                return self._send(CAMP.export_next(), "text/csv", download="next_run.csv")
            if path == "/api/progress":
                return self._json(dict(progress=CAMP.progress))
            if path == "/api/finder/examples":
                return self._json(FINDER.examples())
            if path == "/api/finder/status":
                return self._json(FINDER.status())
            if path.startswith("/examples/discovery/"):
                name = os.path.basename(path)
                ctype = "text/csv" if name.endswith(".csv") else "text/plain; charset=utf-8"
                return self._send(FINDER.example_file(name), ctype, download=name)
            name = os.path.basename(path)
            fp = os.path.join(UI_DIR, name)
            ext = os.path.splitext(name)[1]
            if ext in MIME and os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    return self._send(fh.read(), MIME[ext])
            self._send("not found", "text/plain", 404)
        except ValueError as e:
            self._json(dict(error=str(e)), 400)
        except Exception as e:                      # keep the page alive whatever happens
            self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return self._json(dict(error="bad JSON"), 400)
        fn = ROUTES.get(path)
        if fn is None:
            return self._send("not found", "text/plain", 404)
        try:
            t0 = time.time()
            out = fn(body)
            if isinstance(out, dict) and "analysis" in out:
                out["analysis"]["seconds"] = round(time.time() - t0, 1)
            self._json(out)
        except ValueError as e:
            self._json(dict(error=str(e)), 400)
        except Exception as e:
            self._json(dict(error=f"{type(e).__name__}: {e}"), 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"Reactor Advisor is running at {url}   (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
