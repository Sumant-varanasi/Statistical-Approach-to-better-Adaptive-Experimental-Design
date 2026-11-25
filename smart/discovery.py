"""
Reaction discovery: working out WHICH reactions happen in the bed from the GC readings.

The advisor (advisor.py) is told the reaction network and learns only the rate constants.
This module removes that assumption.  It answers two questions an operator or chemist asks
before any optimisation:

  1. "I have two or three ideas of what is going on.  Which one do my data support?"
         rank_candidates(candidates, data)
  2. "I do not know the reactions at all.  What do the data say?"
         discover(data)                    automatic search over a library of possible reactions
     and, when two answers fit equally well,
         discriminating_run(...)           the next experiment that would tell them apart

How it works, in plain words
----------------------------
* A candidate network is a short list of reactions typed as text, one per line:
        A + B <=> C
        C + B -> E
        A -> D ; orders A=2
  '->' runs one way, '<=>' runs both ways and stops at the equilibrium.  By default the rate
  follows the equation (mass action: A + 2 B -> E has r = k P_A P_B^2); 'orders' overrides the
  exponent of any species (order 0 = the rate does not depend on that gas while it is present).
* Every network is turned into a reactor model (the same fixed-bed model as the advisor,
  systems.ReactionSystem) and its rate constants and activation energies are fitted to all
  the GC mole fractions (kinetics.KineticFit).
* How well a network explains the data is measured in units of the GC's own noise
  (1 % of the reading + 0.05 mol-%):   chi2 = sum ((model - GC) / noise)^2.
  A network that is right leaves misses about as large as the noise.
* Networks are compared by BIC = chi2 + (number of fitted parameters) x ln(number of data):
  an extra reaction must buy a clear improvement of the fit, otherwise the simpler network
  wins.  Two networks whose BIC differs by less than INDISTINGUISHABLE (6) are reported as
  "the data cannot tell these apart", never as a decision.
* The automatic search builds a network one reaction at a time from a library that is
  ENUMERATED from the atom balance (every balanced reaction among the species with small
  coefficients), not written around a known answer.  Chemistry prunes the search: a reaction
  can only be added if its reactants are fed or made by the reactions chosen so far, a
  product the GC never sees is not proposed, and directions that thermodynamics rules out
  over the whole operating box are left out.  The search changes one thing at a time (add,
  remove or replace a reaction, or change one reaction order) and keeps the best change, so
  it is not guaranteed to find the best network there is; when what it ends with does not
  fit within the GC noise it starts again from another first reaction and says so.  It is
  quickest and most reliable when the true rates follow the equations (mass action).  A
  true rate that is, for example, zero order in a reactant is harder: the first pass takes
  every rate to follow its equation until the reactions are settled, and in trials with
  four other run designs that pass and one new start missed such a network three times out
  of four.  From its second new start on the search therefore also tries every reaction it
  may add with one of its orders changed; with that, all four were found, at the price of
  2 to 3 minutes instead of about 1.  That is a handful of trials with one kind of unusual
  rate law, not a guarantee; a miss is reported as "does not fit within the GC noise", not
  as an answer.  Ranking typed candidates has no such limit: it compares exactly the
  networks given.
* "Within the GC noise" is judged against the advisor's noise model, 1 % of the reading PLUS
  0.05 mol-%.  The virtual reactor draws the two parts independently, so the real scatter is
  up to 1.4 times smaller than that sum and a correct network typically misses by 0.5 to 0.8
  "times the noise", not 1.  The test for "fits" is therefore lenient by about that factor,
  and BIC differences between networks come out somewhat smaller than with the exact scatter
  (the cautious side for calling a decision).

Which model classes of the kinetic fit are allowed here, and why
----------------------------------------------------------------
kinetics.KineticFit has a ladder: plain, inhib, inhib_T, decay, plain_decay.  Discovery uses
  'plain'   power-law rates, 2 parameters per reaction (ln k, Ea)
  'inhib'   + one parameter: product C blocking catalyst sites (every rate / (1 + K_C P_C)^m)
and not the others.  'inhib' is allowed because site blocking is common on real catalysts
and, if it is ignored, a network with a WRONG extra reaction can win simply by imitating
it.  The catalyst-ageing classes ('decay', 'plain_decay') describe how the catalyst changes
with time on stream, not which reactions happen, and they need a repeated set point to be
seen at all; 'inhib_T' adds a second adsorption parameter that about 30 runs rarely pin down.
The automatic search runs with 'plain' only (speed) and re-fits its best few networks with
'inhib' allowed at the end.  Pass models=("plain",) for pure power-law kinetics.

Species
-------
The species are those of the complex system (systems.COMPLEX): A and B are fed, I is the
inert, C = AB, D = A rearranged (one A unit), E = AB2.  SPECIES_TABLE holds their
composition (for the atom balance) and PLACEHOLDER formation enthalpies and entropies, chosen
to reproduce COMPLEX.thermo (A + B <=> C: -120 kJ/mol, -170 J/mol/K) and COMPLEX.dH, and so
that the two reactions COMPLEX writes as one-way (C + B -> E and A -> D) are indeed complete
(K > 100) over the whole operating box.  From the table the equilibrium constant of ANY
balanced reaction follows:  K = exp(-dH/RT + dS/R).
Be aware what that choice does.  The table was fitted to the complex example's answer, and the
thermodynamic pruning of the library (reaction_library) then removes exactly the other forms of
those two reactions ('D -> A', 'D <=> A', 'E -> B + C', 'B + C <=> E'): it encodes the
directions of that answer.  Making A -> D complete also needs D to have an entropy 160 J/mol/K
above A's, more than an isomerisation would have.  With the unpruned library
(reaction_library(thermo_prune=False)) the search still finds the same reactions on the
example files, but then cannot tell those two one-way steps from their two-way forms (which
predict the same inside the box) and says so.  With real species the table must come from
measured or tabulated thermodynamics.

ALL NUMBERS ARE PLACEHOLDERS, like the rest of the twin.
"""
import csv
import io
import math
import os
import re
import time
from itertools import combinations_with_replacement
import numpy as np
from scipy.stats import chi2 as _chi2_dist, qmc
from .systems import (ReactionSystem, COMPLEX, A_SPEC, R, T_REF, LOW, HIGH, MAX_STEP, MIN_TOTAL_FLOW,
                      theta_from, _solve, _GAMMA)
from .kinetics import (KineticFit, GC_REL, GC_ABS, GC_NOISE_REL, GC_NOISE_ABS, HOURS_PER_RUN, PRIOR_E,
                       PRIOR_SD_A, PRIOR_SD_B)

# ------------------------------------------------------------------ constants of the method
INDISTINGUISHABLE = 6.0     # BIC difference below which two networks are "not told apart" (Kass & Raftery: < 6 is not strong evidence)
CLEAR_GAP = 10.0            # BIC difference from which "no other network comes close" is said (Kass & Raftery: > 10 is very strong)
MIN_GAIN = 2.0              # a search move is accepted only if it lowers BIC by more than this
MISS_LIMIT = 2.0            # a species is "not explained" when its typical miss is more than 2x the GC noise
FIT_LEVEL = 0.01            # "fits within the GC noise" = passes a chi-square test at the 1 % level
FEW_RUNS = 6                # below this number of runs every result carries a caution (5 screening runs are not a kinetic study)
GENERIC_K, GENERIC_EA = 1e-6, 100e3    # starting guess of every rate constant (per gram at 100 m2/g) and Ea (J/mol)
ZERO_ORDER_CUT = 1e-3       # a zero-order reactant stops driving the rate below 0.1 mol-% (it has run out)
K_COMPLETE = 100.0          # a direction is "complete everywhere" if its equilibrium constant exceeds this over the whole box
MODELS = ("plain", "inhib")
ORDER_CHOICES = (0, 1, 2)
N_UNCERTAIN = 256           # discriminating_run: how many of its best points are scored again with parameter uncertainty

SPECIES_TABLE = {
    #        composition          formation enthalpy (J/mol) and entropy (J/mol/K): PLACEHOLDERS
    "A": dict(atoms={"A": 1}, H=0.0, S=300.0, role="fed"),
    "B": dict(atoms={"B": 1}, H=0.0, S=200.0, role="fed"),
    "C": dict(atoms={"A": 1, "B": 1}, H=-120e3, S=330.0, role="product"),     # A + B <=> C: -120 kJ/mol, -170 J/mol/K
    "D": dict(atoms={"A": 1}, H=50e3, S=460.0, role="product"),               # A -> D: +50 kJ/mol, complete over the box
    "E": dict(atoms={"A": 1, "B": 2}, H=-180e3, S=510.0, role="product"),     # C + B -> E: -60 kJ/mol, complete over the box
    "I": dict(atoms={}, H=0.0, S=0.0, role="inert"),
}
LOG_COLS = ["pressure_bar", "flow_A_mL_min", "flow_B_mL_min", "flow_I_mL_min", "temperature_C", "catalyst_g"]


class ReactionError(ValueError):
    """A reaction or network that cannot be used, with a message in plain words."""


# =================================================================================== reactions as text
_ARROWS = (("<=>", True), ("<-->", True), ("<->", True), ("⇌", True), ("-->", False), ("=>", False), ("->", False), ("→", False))
_TERM = re.compile(r"^(\d*)\s*\*?\s*([A-Za-z][A-Za-z0-9_]*)$")


def _num(v):
    return f"{v:g}"


def _side_text(side):
    return " + ".join((f"{c} " if c != 1 else "") + s for s, c in side.items())


def _power(orders, species):
    """'P_A P_B^2' for a dict of orders; '1' when every order is zero."""
    out = [f"P_{s}" + (f"^{_num(orders[s])}" if orders[s] != 1 else "") for s in species if orders.get(s, 0) > 0]
    return " ".join(out) if out else "1"


class Reaction:
    """One reaction: what reacts, what is made, one-way or both ways, and the rate law.

    reactants, products   {species: whole-number coefficient}, in the order they were written
    reversible            True for '<=>' (the reaction stops at its equilibrium)
    orders                {species: exponent of its partial pressure in the FORWARD rate}
                          default = the coefficient of every reactant (mass action)
    For a reversible reaction the reverse rate follows from thermodynamics: its orders are the
    forward orders plus the stoichiometric coefficients, so the rate is zero exactly at the
    equilibrium composition:  r = k (forward term - reverse term / K)."""

    def __init__(self, reactants, products, reversible=False, orders=None, table=None):
        table = SPECIES_TABLE if table is None else table
        self.reactants = {s: int(c) for s, c in reactants.items()}
        self.products = {s: int(c) for s, c in products.items()}
        self.reversible = bool(reversible)
        full = {s: float(c) for s, c in self.reactants.items()}
        full.update({s: float(v) for s, v in (orders or {}).items()})
        self.orders = {s: v for s, v in full.items() if v != 0 or s in self.reactants}
        self._check(table)

    # ------------------------------------------------------------------ checks, with plain-word messages
    def _check(self, table):
        eq = self.equation()
        if not self.reactants or not self.products:
            raise ReactionError(f"'{eq}': a reaction needs at least one species on each side of the arrow.")
        names = [s for s in table if table[s]["role"] != "inert"]
        inert = [s for s in table if table[s]["role"] == "inert"]
        for s in list(self.reactants) + list(self.products) + list(self.orders):
            if s in inert:
                raise ReactionError(f"'{eq}': {s} is the inert gas; it does not react and cannot appear in a reaction.")
            if s not in table:
                raise ReactionError(f"Unknown species '{s}' in '{eq}'. The species are {', '.join(names)} "
                                    f"(and {', '.join(inert)}, the inert gas, which does not react).")
        both = [s for s in self.reactants if s in self.products]
        if both:
            raise ReactionError(f"'{eq}': {both[0]} appears on both sides of the arrow; write the net reaction.")
        left, right = atom_count(self.reactants, table), atom_count(self.products, table)
        if left != right:
            def say(c):
                return ", ".join(f"{_num(v)} {a}-unit" + ("s" if v != 1 else "") for a, v in sorted(c.items())) or "nothing"
            raise ReactionError(f"'{eq}' is not balanced: the left side holds {say(left)}, the right side {say(right)}. "
                                "(A and D are one A-unit, B is one B-unit, C = AB, E = AB2.)")
        for s, v in self.orders.items():
            if not (0 <= v <= 3):
                raise ReactionError(f"'{eq}': the order of {s} must be between 0 and 3 (got {_num(v)})."
                                    + (f" The order is taken from the coefficient {_num(v)} of {s}; write the rate you mean "
                                       f"after the equation, for example '; orders {s}=1'."
                                       if self.reactants.get(s) == v and v > 3 else ""))
        if not any(v > 0 for v in self.orders.values()):
            raise ReactionError(f"'{eq}': every order is zero, so the rate would not depend on any gas. "
                                "Give at least one species a positive order.")
        if self.reversible:
            low = [s for s, c in self.reactants.items() if self.orders.get(s, 0) < c]
            if low:
                raise ReactionError(f"'{eq}': in a reversible reaction the order of a reactant cannot be below its "
                                    f"coefficient ({low[0]}), because the reverse rate would become infinite when "
                                    f"{low[0]} runs out. Use '->' or a higher order.")

    # ------------------------------------------------------------------ text
    def equation(self):
        """The equation alone, e.g. 'A + B <=> C' (no rate law)."""
        return f"{_side_text(self.reactants)} {'<=>' if self.reversible else '->'} {_side_text(self.products)}"

    @property
    def default_orders(self):
        return self.orders == {s: float(c) for s, c in self.reactants.items()}

    def text(self):
        """The reaction in the text format parse_reaction reads."""
        if self.default_orders:
            return self.equation()
        show = list(self.reactants) + [s for s in self.orders if s not in self.reactants]
        return self.equation() + " ; orders " + ", ".join(f"{s}={_num(self.orders.get(s, 0))}" for s in show)

    def reverse_orders(self):
        """Orders of the reverse rate of a reversible reaction (forward orders + coefficients)."""
        rev = dict(self.orders)
        for s, c in self.reactants.items():
            rev[s] = rev.get(s, 0.0) - c
        for s, c in self.products.items():
            rev[s] = rev.get(s, 0.0) + c
        return {s: v for s, v in rev.items() if abs(v) > 1e-12}

    def rate_law(self, j=None):
        """'r1 = k1 (P_A P_B - P_C / K1)' and so on; j = number of the reaction in its network."""
        n = "" if j is None else str(j)
        order = list(SPECIES_TABLE)
        if self.reversible:
            return f"r{n} = k{n} ({_power(self.orders, order)} - {_power(self.reverse_orders(), order)} / K{n})"
        zero = [s for s in self.reactants if self.orders.get(s, 0) == 0]
        return f"r{n} = k{n} {_power(self.orders, order)}" + (f"   (zero order in {', '.join(zero)})" if zero else "")

    def k_units(self):
        """Units of the rate constant as reported (per gram of catalyst), e.g. 'mol g^-1 s^-1 bar^-2'."""
        n = sum(self.orders.values())
        return "mol g^-1 s^-1" + ("" if n == 0 else " bar^-1" if n == 1 else f" bar^-{_num(n)}")

    # ------------------------------------------------------------------ identity
    def _sides(self):
        f = lambda side: "+".join((str(c) if c != 1 else "") + s for s, c in sorted(side.items()))
        return f(self.reactants), f(self.products)

    def key(self):
        """Canonical name: the same reaction written in a different order has the same key."""
        l, r = self._sides()
        o = "" if self.default_orders else ";" + ",".join(f"{s}={_num(v)}" for s, v in sorted(self.orders.items()))
        return f"{l}{'<=>' if self.reversible else '->'}{r}{o}"

    def entry_key(self):
        """The same for a reaction and its reverse, whatever the orders: the library entry."""
        return " = ".join(sorted(self._sides()))

    def direction_key(self):
        """Entry + which side reacts: a rate constant can be carried over between two reactions
        with the same direction_key (one-way <-> reversible, changed orders)."""
        return self._sides()[0] + ">" + self._sides()[1]

    def __eq__(self, other):
        return isinstance(other, Reaction) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return f"Reaction('{self.text()}')"

    # ------------------------------------------------------------------ numbers
    def thermo(self, table=None):
        """(dH J/mol, dS J/mol/K) of the reaction as written, from the species table."""
        table = SPECIES_TABLE if table is None else table
        dH = sum(c * table[s]["H"] for s, c in self.products.items()) - sum(c * table[s]["H"] for s, c in self.reactants.items())
        dS = sum(c * table[s]["S"] for s, c in self.products.items()) - sum(c * table[s]["S"] for s, c in self.reactants.items())
        return float(dH), float(dS)

    def reversed(self):
        """The one-way reaction the other way round (products react), with mass-action orders."""
        return Reaction(self.products, self.reactants, False)


def atom_count(side, table=None):
    """{atom: number} of one side of a reaction."""
    table = SPECIES_TABLE if table is None else table
    out = {}
    for s, c in side.items():
        for a, n in table[s]["atoms"].items():
            out[a] = out.get(a, 0) + c * n
    return out


def parse_reaction(text, table=None):
    """'A + B <=> C', 'C + B -> E ; orders C=1, B=0' -> Reaction.  Raises ReactionError with a
    plain-word message for bad syntax, unknown species or unbalanced atoms."""
    table = SPECIES_TABLE if table is None else table
    raw = str(text).split("#")[0].strip()
    if not raw:
        raise ReactionError("Empty line: write a reaction such as 'A + B <=> C'.")
    eq, _, opt = raw.partition(";")
    arrow = next(((a, rev) for a, rev in _ARROWS if a in eq), None)
    if arrow is None:
        raise ReactionError(f"Could not read '{raw}': a reaction needs an arrow, '->' (one way) or '<=>' "
                            "(both ways), for example 'A + B <=> C'.")
    lhs, _, rhs = eq.partition(arrow[0])

    def side(s, where):
        out = {}
        if not s.strip():
            raise ReactionError(f"'{raw}': nothing is written {where} the arrow. A reaction needs at least one species on "
                                "each side, for example 'A + B <=> C'.")
        for term in s.split("+"):
            m = _TERM.match(term.strip())
            if not m:
                raise ReactionError(f"Could not read '{term.strip() or s.strip()}' in '{raw}'. Write each side as "
                                    "species joined by '+', with whole-number coefficients, for example 'A + 2 B -> E'.")
            c, name = int(m.group(1) or 1), m.group(2)
            name = name if name in table else name.upper()
            if c < 1:
                raise ReactionError(f"'{raw}': a coefficient must be 1 or more.")
            out[name] = out.get(name, 0) + c
        return out

    orders = {}
    opt = opt.strip()
    if opt:
        m = re.match(r"^orders?\b[:\s]*(.*)$", opt, re.I)
        if not m:
            raise ReactionError(f"Could not read '{opt}' in '{raw}'. After the ';' write the orders, for example "
                                "'; orders A=2' or '; orders C=1, B=0'.")
        for item in re.split(r"[,\s]+", m.group(1).replace(" =", "=").replace("= ", "=").strip()):
            if not item:
                continue
            name, eqs, val = item.partition("=")
            try:
                if not name:
                    raise ValueError(item)
                orders[name if name in table else name.upper()] = float(val)
            except ValueError:
                raise ReactionError(f"Could not read the order '{item}' in '{raw}'. Write it as 'A=2'.") from None
        if not orders:
            raise ReactionError(f"'{raw}': 'orders' is given but no order follows. Write for example '; orders A=2'.")
    return Reaction(side(lhs, "before"), side(rhs, "after"), arrow[1], orders, table)


def parse_network(source, table=None, unique=True):
    """A network from text (one reaction per line; blank lines and '#' comments are ignored), a
    list of lines, a list of Reaction objects, or a system built by build_system.
    unique=False skips the check that no reaction is listed twice (used internally)."""
    if isinstance(source, ReactionSystem):
        return list(getattr(source, "reactions", None) or reactions_from_system(source))
    if isinstance(source, Reaction):
        return [source]
    lines = source.splitlines() if isinstance(source, str) else list(source)
    out = []
    for ln in lines:
        if isinstance(ln, Reaction):
            r = ln
        elif not str(ln).split("#")[0].strip():
            continue
        else:
            r = parse_reaction(ln, table)
        if unique and any(r.entry_key() == q.entry_key() for q in out):
            raise ReactionError(f"'{r.equation()}' is listed twice (a reaction and its reverse count as the same "
                                "reaction; use '<=>' for a reaction that runs both ways).")
        out.append(r)
    return out


def read_network_file(path):
    """A network from a text file (one reaction per line; Windows line ends and a byte-order mark are fine)."""
    with open(path, encoding="utf-8-sig") as fh:
        return parse_network(fh.read())


def network_text(reactions):
    """The network in the text format, one reaction per line (round-trips through parse_network)."""
    return [r.text() for r in parse_network(reactions)]


def format_network(reactions):
    """Readable lines: number, equation and rate law, e.g. '1.  A + B <=> C     r1 = k1 (P_A P_B - P_C / K1)'."""
    net = parse_network(reactions)
    w = max([len(r.equation()) for r in net] + [1])
    return [f"{j}.  {r.equation():<{w}}    {r.rate_law(j)}" for j, r in enumerate(net, 1)]


def network_key(reactions):
    """Canonical name of a network (a list of Reaction): the same reactions and rate laws in any order give the same key."""
    return " | ".join(sorted(r.key() for r in reactions))


def reactions_from_system(S):
    """Reaction objects read back from any ReactionSystem (its stoichiometry and orders)."""
    out = []
    for j in range(S.nr):
        re_ = {s: int(round(-S.nu[i, j])) for i, s in enumerate(S.species) if S.nu[i, j] < 0}
        pr = {s: int(round(S.nu[i, j])) for i, s in enumerate(S.species) if S.nu[i, j] > 0}
        orders = {s: float(S.fwd[j, i]) for i, s in enumerate(S.species) if S.fwd[j, i] != 0 or s in re_}
        out.append(Reaction(re_, pr, S.thermo[j] is not None, orders))
    return out


# =================================================================================== networks as reactor models
class NetworkSystem(ReactionSystem):
    """A ReactionSystem built from Reaction objects.  One addition to the parent's physics: a
    reactant of ORDER ZERO does not influence the rate while it is present, but the reaction
    must stop when that reactant has run out, otherwise its flow would become negative.  The
    rate is therefore multiplied by y / (y + 0.001), y = mole fraction of that reactant: 1 for
    all practical purposes, falling to zero below about 0.1 mol-% (the GC detection level).
    Without zero-order reactants the physics is exactly that of the parent class.

    Speed.  Discovery fits hundreds of networks, and nearly all the time goes into integrating
    the bed.  simulate(method='fast') therefore uses the parent's Rosenbrock scheme (same
    steps, same tolerances) but takes the derivative of the rates with respect to the
    composition from the rate laws themselves (a power law is easy to differentiate) instead
    of by finite differences: 2 rate evaluations per step instead of 8 (see _bed).  The result
    agrees with the parent's fast solver to better than 1e-3 of the inlet flow and with the
    accurate solver (LSODA) as closely as the parent's does (far below the GC noise inside the
    study window; checked in test_discovery.py).  method='lsoda' and truth=True are passed to
    the parent unchanged."""

    reactions = ()
    _zero = ()

    def rates(self, Pp, T, theta, inhib=0.0, K_A=0.0):
        """The parent's rates (n x nr), with every reaction that has a zero-order reactant stopped as
        that reactant runs out (see the class note)."""
        r = super().rates(Pp, T, theta, inhib, K_A)
        if self._zero:
            cut = ZERO_ORDER_CUT * Pp.sum(1)
            for j, i in self._zero:
                r[:, j] = r[:, j] * Pp[:, i] / (Pp[:, i] + cut)
        return r

    def simulate(self, P, VA, VB, VI, W, T, theta=None, truth=False, method="fast", groups=None,
                 a_spec=A_SPEC, inhib=None):
        """Outlet and inlet molar flows (n x ns, mol/s); same call and meaning as the parent."""
        if truth or method != "fast":
            return super().simulate(P, VA, VB, VI, W, T, theta=theta, truth=truth, method=method, groups=groups,
                                    a_spec=a_spec, inhib=inhib)
        P, VA, VB, VI, W, T, a_s = (np.atleast_1d(np.asarray(v, float)) for v in (P, VA, VB, VI, W, T, a_spec))
        n = max(map(len, (P, VA, VB, VI, W, T, a_s)))
        P, VA, VB, VI, W, T, a_s = (np.broadcast_to(v, (n,)).copy() for v in (P, VA, VB, VI, W, T, a_s))
        F0 = self.inlet_flows(VA, VB, VI)
        FT0 = F0.sum(1)
        sc = a_s * W / FT0                                       # catalyst surface per molar feed
        th = np.atleast_2d(np.asarray(theta, float))
        lnk = np.broadcast_to(th[:, 0::2] - th[:, 1::2] * (T_REF / T[:, None] - 1.0), (n, self.nr))
        with np.errstate(invalid="ignore"):
            lnk_eq = self.log_keq(T)
            KP = None
            if inhib is not None and np.any(inhib):              # product C on the sites: every rate / (1 + K_C P_C)^m
                KP = (np.broadcast_to(np.asarray(inhib, float), (n,)) * P)[:, None]
            zero = np.zeros((1, self.nr, self.ns))
            for j, i in self._zero:
                zero[0, j, i] = 1.0
            f = _bed(self.fwd[None], self.rev[None], self.nu[None], zero if self._zero else None, lnk,
                     lnk_eq if np.isfinite(lnk_eq[0]).any() else None, np.log(P), sc, F0 / FT0[:, None],
                     KP=KP, iC=self.iC, groups=groups)
            f = np.where(np.isnan(f), np.nan, np.maximum(f, 0.0))
        return f * FT0[:, None], F0


def _bed(fwd, rev, nu, zero, lnk, lnk_eq, lnP, sc, f0, KP=None, iC=2, groups=None):
    """Integrate the mole balances along the bed for many rows (experiments) at once.

    The rows may belong to ONE network (fwd, rev, nu, zero have a leading axis of length 1) or
    to DIFFERENT networks (leading axis = rows; each row carries the orders and the
    stoichiometry of its own network, padded with empty reactions to the same number).  The
    search uses the second form to fit all the networks of a step in one integration.

    fwd, rev   forward / reverse orders (1 or rows, nr, ns);  nu  stoichiometry (1 or rows, ns, nr)
    zero       1 where a reactant has order zero (same shape as fwd), or None
    lnk        ln k_j at the row's temperature (rows, nr); -inf switches a padded reaction off
    lnk_eq     ln K_eq (rows, nr), +inf for one-way reactions; None if no reaction is reversible
    lnP        ln of the pressure (rows,);  sc  catalyst surface per molar feed (rows,)
    f0         inlet flows divided by the total inlet flow (rows, ns)
    KP         K_C x P (rows, 1) for site blocking by product C (species iC), or None

    Rate of reaction j:  r_j = (exp(cf_j + fwd_j . ln y) - exp(cb_j + rev_j . ln y)) x factors,
    y = mole fractions, cf = ln k + (sum of orders) ln P.  Because this is a sum of power laws,
    its derivative with respect to the composition is written down directly:
    d r / d ln y = forward term x fwd - reverse term x rev (+ the terms of the factors)."""
    lnP = lnP[:, None]
    m_j = fwd.sum(2)                                             # sites in the forward step
    cf = lnk + m_j * lnP
    cb = None if lnk_eq is None else lnk - lnk_eq + rev.sum(2) * lnP
    cur = dict(n=-1)

    def pick(X, idx):
        return X if X is None or len(X) == 1 else X[idx]

    def rhs(f, idx, jac=False):
        """df/ds for the rows idx and, with jac, its derivative with respect to f (rows, ns, ns)."""
        if len(idx) != cur["n"]:               # rows only ever LEAVE the active set: same number = same rows
            cur.update(n=len(idx), cf=cf[idx], cb=pick(cb, idx), sc=sc[idx], fwd=pick(fwd, idx), rev=pick(rev, idx),
                       nu=pick(nu, idx), zero=pick(zero, idx), m=pick(m_j, idx), KP=pick(KP, idx))
        f = np.maximum(f, 0.0)
        s = f.sum(1, keepdims=True)
        y = np.maximum(f / s, 1e-30)
        ly = np.log(y)[:, :, None]
        fw = np.exp(cur["cf"] + np.matmul(cur["fwd"], ly)[:, :, 0])
        r = fw
        if cb is not None:
            bk = np.exp(cur["cb"] + np.matmul(cur["rev"], ly)[:, :, 0])
            r = fw - bk
        phi = None                                               # factors that multiply the power laws
        if zero is not None:                                     # a zero-order reactant that has run out stops its reaction
            phi = np.exp(np.matmul(cur["zero"], np.log(y / (y + ZERO_ORDER_CUT))[:, :, None])[:, :, 0])
        if KP is not None:
            site = 1.0 + cur["KP"] * y[:, [iC]]
            q = site ** -cur["m"]
            phi = q if phi is None else phi * q
        if phi is not None:
            r = r * phi
        g = cur["sc"][:, None] * np.matmul(cur["nu"], r[:, :, None])[:, :, 0]
        if not jac:
            return g
        D = fw[:, :, None] * cur["fwd"]                          # d r / d ln y, (rows, nr, ns)
        if cb is not None:
            D -= bk[:, :, None] * cur["rev"]
        if phi is not None:
            D *= phi[:, :, None]
            if zero is not None:
                D += r[:, :, None] * (cur["zero"] * (ZERO_ORDER_CUT / (y + ZERO_ORDER_CUT))[:, None, :])
            if KP is not None:
                D[:, :, iC] -= r * cur["m"] * (cur["KP"] * y[:, [iC]] / site)
        G = np.matmul(cur["nu"], D / y[:, None, :])              # d (nu r) / d y
        J = (G - np.matmul(G, y[:, :, None])) * (cur["sc"] / s[:, 0])[:, None, None]       # y = f / sum(f)
        return g, J

    return _ros2_analytic(rhs, f0, groups)


def _ros2_analytic(rhs, f0, groups=None, rtol=3e-3, atol=3e-5, h0=1e-9, max_iter=5000):
    """systems._ros2 (second-order L-stable Rosenbrock method, one step size per experiment,
    error estimate against linearly-implicit Euler, same tolerances) with the Jacobian supplied
    by rhs(y, rows, jac=True) instead of built by finite differences."""
    n, ns = f0.shape
    f, s = f0.copy(), np.zeros(n)
    h = np.full(n, h0)
    eye = np.eye(ns)
    for _ in range(max_iter):
        act = np.nonzero(s < 1.0 - 1e-12)[0]
        if len(act) == 0:
            break
        y = f[act]
        hh = np.minimum(h[act], 1.0 - s[act])
        g0, J = rhs(y, act, True)
        if not np.isfinite(J).all():
            J[~np.isfinite(J)] = 0.0
        if not np.isfinite(g0).all():
            g0 = np.nan_to_num(g0, nan=0.0, posinf=0.0, neginf=0.0)
        M = eye - (_GAMMA * hh)[:, None, None] * J
        k1 = _solve(M, g0)
        k2 = _solve(M, rhs(y + hh[:, None] * k1, act) - 2.0 * k1)
        y_new = y + hh[:, None] * (1.5 * k1 + 0.5 * k2)
        y_eul = y + hh[:, None] * _solve(eye - hh[:, None, None] * J, g0)
        ratio = (np.abs(y_new - y_eul) / (atol + rtol * np.abs(y_new))).max(1)
        ratio[(y_new < -atol).any(1) | ~np.isfinite(ratio)] = 1e3       # reject negative / NaN steps
        if groups is not None:                       # an experiment and its perturbed copies share steps
            gmax = np.zeros(groups.max() + 1)
            np.maximum.at(gmax, groups[act], ratio)
            ratio = gmax[groups[act]]
        ok = ratio <= 1.0
        f[act[ok]] = y_new[ok]
        s[act[ok]] += hh[ok]
        h[act] = hh * np.clip(0.9 / np.sqrt(np.maximum(ratio, 1e-10)), 0.2, 5.0)
    else:
        f[s < 1.0 - 1e-12] = np.nan                  # absurd trial kinetics: no result, not a crash
    return f


def generic_theta(nr, a_spec=A_SPEC):
    """The starting guess used for every reaction of a typed or discovered network: the same
    order of magnitude as theta_guess of the built-in systems (k = 1e-6 per gram, Ea = 100 kJ/mol)."""
    return theta_from([GENERIC_K] * nr, [GENERIC_EA] * nr, a_spec)


def build_system(reactions, base=COMPLEX, name="network", theta=None, table=None, unique=True):
    """Reactions (text, lines or Reaction objects) -> a reactor model (NetworkSystem).

    base     the system that supplies the species list, the prices and the wanted product
    theta    kinetics [ln k_1, Ea_1/(R T_ref), ...] to store as theta_true (used when the network
             is the 'truth' of an example); default = the generic starting guess
    Reversible reactions get their equilibrium constant from the species table."""
    table = SPECIES_TABLE if table is None else table
    net = parse_network(reactions, table, unique)
    if not net:
        raise ReactionError("The network is empty: give at least one reaction.")
    sp = list(base.species)
    for r in net:
        for s in list(r.reactants) + list(r.products) + list(r.orders):
            if s not in sp:
                raise ReactionError(f"'{r.equation()}' uses {s}, which the '{base.name}' system does not have "
                                    f"(its species are {', '.join(sp)}).")
    nr = len(net)
    nu, fwd, rev = np.zeros((len(sp), nr)), np.zeros((nr, len(sp))), np.zeros((nr, len(sp)))
    thermo, dH = [], []
    for j, r in enumerate(net):
        for s, c in r.reactants.items():
            nu[sp.index(s), j] = -c
        for s, c in r.products.items():
            nu[sp.index(s), j] = c
        for s, v in r.orders.items():
            fwd[j, sp.index(s)] = v
        h, s_ = r.thermo(table)
        dH.append(h)
        if r.reversible:
            for s, v in r.reverse_orders().items():
                rev[j, sp.index(s)] = v
            thermo.append((h, s_))
        else:
            thermo.append(None)
    guess = generic_theta(nr)
    S = NetworkSystem(name, sp, nu, fwd, rev, thermo, theta_true=guess if theta is None else theta,
                      theta_guess=guess, K_inhib=0.0, prices=base.prices, product=base.species[base.iC],
                      description=",  ".join(r.equation() for r in net), dH=dH)
    S.reactions = tuple(net)
    S._zero = tuple((j, sp.index(s)) for j, r in enumerate(net) for s in r.reactants if r.orders.get(s, 0) == 0)
    return S


# =================================================================================== the candidate networks of the worked example
TRUTH_NOTE = "A + B <=> C with r = k1 (P_A P_B - P_C / K);  C + B -> E with r = k2 P_C P_B;  A -> D with r = k3 P_A"
CANDIDATES = {
    "truth": "# The reactions the complex example data were calculated from.\n"
             "A + B <=> C\n"
             "C + B -> E\n"
             "A -> D\n",
    "close": "# Quite close: the same three reactions, one order wrong\n"
             "# (the over-reaction is taken as zero order in B: r = k P_C).\n"
             "A + B <=> C\n"
             "C + B -> E ; orders C=1, B=0\n"
             "A -> D\n",
    "similar": "# Relatively the same: three reactions, but a different pathway and no equilibrium\n"
               "# (E is made directly from the feed, in parallel, with r = k P_A P_B).\n"
               "A + B -> C\n"
               "A + 2 B -> E ; orders A=1, B=1\n"
               "A -> D\n",
    "far": "# Very different: only two reactions, different orders, nothing makes E, no equilibrium\n"
           "# (C is made at a rate that ignores B; D comes from C falling apart).\n"
           "A + B -> C ; orders A=1, B=0\n"
           "C -> D + B\n",
}
CANDIDATE_WORDS = {"truth": "the true network", "close": "quite close (one order wrong)",
                   "similar": "relatively the same (different pathway, no equilibrium)",
                   "far": "very different (two reactions, other orders, nothing makes E)"}


def candidate_system(name):
    """One of the four example candidates ('truth', 'close', 'similar', 'far') as a reactor model."""
    if name not in CANDIDATES:
        raise ReactionError(f"Unknown candidate '{name}'. Choose from {', '.join(CANDIDATES)}.")
    return build_system(CANDIDATES[name], base=COMPLEX, name=name)


# =================================================================================== data
def make_data(phys, gc, hours=None, a_spec=A_SPEC, name="data", notes=None):
    """Bundle a run log for this module.
    phys   (n, 6): pressure bar, flows of A, B, inert in mL/min, temperature in KELVIN, catalyst g
    gc     (n, 6): mole fractions of A, B, C, D, E, I (fractions or %; rows are renormalised);
           (n, 5) is read as A, B, C, D, I of the simple system (no E)
    notes  plain-word remarks about the log (load_data adds them: a missing column, a run outside
           the operating box); they are repeated in every summary"""
    phys = np.atleast_2d(np.asarray(phys, float))
    gc = np.atleast_2d(np.asarray(gc, float))
    if gc.shape[1] == 5:
        gc = np.insert(gc, 4, 0.0, axis=1)
    if phys.shape[1] != 6 or gc.shape[1] != len(COMPLEX.species) or len(phys) != len(gc):
        raise ValueError("phys must be (n, 6) [P, V_A, V_B, V_I, T, Wcat] and gc (n, 6) [A, B, C, D, E, I]")
    if np.median(gc.sum(1)) > 1.5:
        gc = gc / 100.0
    gc = gc / gc.sum(1, keepdims=True)
    hours = np.full(len(phys), HOURS_PER_RUN) if hours is None else np.asarray(hours, float)
    return dict(phys=phys, gc=gc, hours=hours, a_spec=float(a_spec), species=list(COMPLEX.species), name=name,
                notes=list(notes or []))


def _decode(raw):
    """Bytes of a log file -> text.  UTF-8 (with or without a byte-order mark) or, when the file says
    so with its byte-order mark, UTF-16 (Excel's 'Unicode text'); anything else is refused in plain words."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("The file is not plain text in UTF-8, the encoding this tool reads. In Excel, save it with "
                         "'Save as' -> 'CSV UTF-8 (comma delimited)'.") from None


def _runs_list(idx):
    idx = [int(i) + 1 for i in idx]
    return (f"run {idx[0]}" if len(idx) == 1 else "runs " + ", ".join(map(str, idx[:6])) + (" and more" if len(idx) > 6 else ""))


def load_data(source, a_spec=A_SPEC):
    """Read a run log in the CSV format of the website and reactor_advisor.py:
        pressure_bar, flow_A_mL_min, flow_B_mL_min, flow_I_mL_min, temperature_C, catalyst_g,
        gc_A, gc_B, gc_C, gc_D, gc_E, gc_I [, hours_h]
    source = a file path, the CSV text itself (or its bytes), or an open file.  Spaces after the
    commas are allowed.  Temperature is converted to kelvin.
    A missing gc_D or gc_E COLUMN is read as zero in every run (the simple system has no E), and
    the data carry a note saying so (repeated in every summary).  An EMPTY cell in a gc_ column
    that is there is refused: a reading left out is not a reading of zero.  Runs outside the
    reactor's operating box are accepted with a note (a temperature typed in kelvin shows as one).
    Every problem is reported as a ValueError in plain words that names the run and the column."""
    name = "data"
    if hasattr(source, "read"):
        text = source.read()
    elif isinstance(source, (bytes, bytearray)):
        text = bytes(source)
    elif "\n" in str(source) or ("," in str(source) and not os.path.exists(str(source))):
        text = str(source)
    elif not str(source).strip():
        raise ValueError("The log is empty: no file name and no CSV text was given.")
    elif not os.path.isfile(str(source)):
        raise ValueError(f"Could not find the file '{source}'.")
    else:
        name = os.path.basename(str(source))
        with open(source, "rb") as fh:
            text = fh.read()
    if isinstance(text, (bytes, bytearray)):
        text = _decode(bytes(text))
    text = text.lstrip("\ufeff")                     # a byte-order mark left in uploaded text would hide the first column
    if not text.strip():
        raise ValueError("The log is empty.")
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    if reader.fieldnames:
        reader.fieldnames = [(f or "").strip() for f in reader.fieldnames]
    rows = [r for r in reader if any((v or "").strip() for v in r.values() if isinstance(v, str))]
    if not rows:
        raise ValueError("No runs found in the file.")
    sp = list(COMPLEX.species)
    need = LOG_COLS + [f"gc_{s}" for s in sp if s not in ("D", "E")]
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise ValueError("The log is missing the columns " + ", ".join(miss) + ". Expected: "
                         + ", ".join(LOG_COLS + [f"gc_{s}" for s in sp]) + " (hours_h is optional)."
                         + (" The columns must be separated by commas." if len(rows[0]) == 1 else ""))
    notes = [f"The log has no gc_{s} column, so {s} is read as zero in every run (right for the simple reaction, which "
             f"makes no E). If the GC does report {s}, add the column: otherwise a network that makes {s} will look wrong."
             for s in ("D", "E") if f"gc_{s}" not in rows[0]]

    def num(i, r, c, default=None):
        """One number of the log, with a plain-word error that names the run and the column."""
        v = (r.get(c) or "").strip()
        if not v and default is not None:
            return default
        try:
            x = float(v)
        except ValueError:
            raise ValueError(f"Run {i} of the log: " + (f"'{v}' in column {c} is not a number." if v else
                                                         f"column {c} is empty."
                                                         + (f" If the GC saw no {c[3:]} in that run, write 0." if c.startswith("gc_") else ""))) from None
        if not np.isfinite(x) or x < 0:
            raise ValueError(f"Run {i} of the log: {v} in column {c} cannot be right (it must be a number of zero or more).")
        return x

    user = np.array([[num(i, r, c) for c in LOG_COLS] for i, r in enumerate(rows, 1)])
    gc = np.array([[num(i, r, f"gc_{s}") if f"gc_{s}" in r else 0.0 for s in sp] for i, r in enumerate(rows, 1)])
    hours = np.array([num(i, r, "hours_h", HOURS_PER_RUN) for i, r in enumerate(rows, 1)])
    empty = np.nonzero(gc.sum(1) <= 0)[0]
    if len(empty):
        raise ValueError(f"Run {empty[0] + 1} of the log has no GC reading (every gc_ column is zero or empty).")
    if np.any(user[:, 1] + user[:, 2] <= 0) or np.any(user[:, 0] <= 0) or np.any(user[:, 5] <= 0):
        i = int(np.nonzero((user[:, 1] + user[:, 2] <= 0) | (user[:, 0] <= 0) | (user[:, 5] <= 0))[0][0]) + 1
        raise ValueError(f"Run {i} of the log: the pressure, the catalyst mass and the feed of A or B must be above zero.")
    hot, cold = np.nonzero(user[:, 4] > HIGH[4] - 273.15 + 1e-6)[0], np.nonzero(user[:, 4] < LOW[4] - 273.15 - 1e-6)[0]
    if len(hot):
        notes.append(f"{_runs_list(hot).capitalize()} of the log {'is' if len(hot) == 1 else 'are'} above the reactor's highest "
                     f"temperature ({HIGH[4] - 273.15:.0f} C): up to {user[hot, 4].max():.0f} C. Was the temperature typed in "
                     "kelvin? The column temperature_C is in degrees Celsius.")
    if len(cold):
        notes.append(f"{_runs_list(cold).capitalize()} of the log {'is' if len(cold) == 1 else 'are'} below the reactor's lowest "
                     f"temperature ({LOW[4] - 273.15:.0f} C).")
    out = [k for k, j in (("pressure", 0), ("flow of A", 1), ("flow of B", 2), ("flow of inert", 3))
           if np.any(user[:, j] > HIGH[j] + 1e-6) or np.any(user[:, j] < LOW[j] - 1e-6)]
    if out:
        notes.append("Some runs have a " + " and a ".join(out) + " outside the reactor's operating box; the fit uses them "
                     "as they are.")
    user[:, 4] += 273.15
    return make_data(user, gc, hours, a_spec, name, notes)


def data_to_csv(data, decimals=6):
    """The log as CSV text (same columns as the website export; temperature in deg C)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run"] + LOG_COLS + [f"gc_{s}" for s in data["species"]] + ["hours_h"])
    for i, (p, g, h) in enumerate(zip(data["phys"], data["gc"], data["hours"]), 1):
        u = list(p[:4]) + [p[4] - 273.15, p[5]]
        w.writerow([i] + [round(float(v), 3) for v in u] + [round(float(v), decimals) for v in g] + [round(float(h), 2)])
    return buf.getvalue()


def gc_noise(y):
    """One standard deviation of a GC mole fraction, as the advisor assumes it (kinetics.py)."""
    return GC_NOISE_REL * np.asarray(y, float) + GC_NOISE_ABS


def fed_species(data):
    """The reactants fed in the log ('A', 'B'): those with a flow above zero in some run."""
    return [s for s, col in (("A", 1), ("B", 2)) if np.any(data["phys"][:, col] > 0)]


def detected_species(data):
    """{species: largest mol-% seen} for the species that are made in the bed: seen by the GC
    clearly above its noise (more than 3 standard deviations) in at least two runs (in the
    only run, if the log holds just one)."""
    out = {}
    for i, s in enumerate(data["species"][:-1]):
        y = data["gc"][:, i]
        if s not in ("A", "B") and np.sum(y > 3 * gc_noise(y)) >= min(2, len(y)):
            out[s] = float(100 * y.max())
    return out


# =================================================================================== fitting one network
def _made_by(net):
    out = set()
    for r in net:
        out |= set(r.products) | (set(r.reactants) if r.reversible else set())
    return out


def _empty_fit(data, name="no reaction"):
    """The 'network' with no reaction at all: the outlet is the feed."""
    S = COMPLEX
    P, VA, VB, VI, T, W = data["phys"].T
    F0 = S.inlet_flows(VA, VB, VI)
    return _score([], None, data, F0 / F0.sum(1, keepdims=True), name, 0.0)


def _one_run(za, n_params=0):
    """Is the misfit concentrated in ONE run?  za: misses per run and species in units of the
    allowance for model error (n, species).  A wrong reaction or rate law shows in every run
    where it matters; a typing error or a bad GC injection shows in one run only.
      worst_run   the run (counted from 1) that carries the largest part of the misfit
      run_share   that part (0..1)
      one_run     True when that run carries more than half of the misfit and more than three
                  times its fair share (at least 3 runs)
      rest_within_allowance   True when the OTHER runs are explained within the allowance (no
                  species beyond it, chi-square per degree of freedom not above 1)"""
    n, k = za.shape
    per_run = np.sum(za ** 2, 1)
    tot = float(per_run.sum())
    i = int(np.argmax(per_run))
    share = float(per_run[i] / tot) if tot > 0 else 0.0
    rest = np.delete(za, i, axis=0)
    rest_ok = bool(n >= 2 and np.all(np.sqrt(np.mean(rest ** 2, 0)) <= 1.0)
                   and np.sum(rest ** 2) / max(rest.size - int(n_params), 1) <= 1.0)
    return dict(worst_run=i + 1, run_share=share, one_run=bool(n >= 3 and share > max(0.5, 3.0 / n)),
                rest_within_allowance=rest_ok)


def _score(net, kf, data, yp, name, seconds):
    """Everything fit_network reports, from the model mole fractions yp (n, ns)."""
    y, sp = data["gc"], data["species"]
    sig = gc_noise(y)
    z = np.nan_to_num((yp - y) / sig, nan=1e3)[:, :-1]
    n_data = int(z.size)
    nr = len(net)
    p = 0 if kf is None else len(kf.theta)
    chi2 = float(np.sum(z ** 2))
    dof = max(n_data - p, 1)
    miss = {}
    for i, s in enumerate(sp[:-1]):
        d = np.nan_to_num(yp[:, i] - y[:, i], nan=1.0)
        miss[s] = dict(mol_pct=float(100 * np.sqrt(np.mean(d ** 2))), x_noise=float(np.sqrt(np.mean(z[:, i] ** 2))),
                       bias_mol_pct=float(100 * np.mean(d)))
    det = detected_species(data)
    made = _made_by(net)
    not_made = [s for s in det if s not in made]
    unexplained = sorted((s for s in miss if miss[s]["x_noise"] > MISS_LIMIT), key=lambda s: -miss[s]["x_noise"])
    fits_noise = bool(chi2 <= _chi2_dist.ppf(1 - FIT_LEVEL, dof)) and not unexplained
    out = dict(name=name, reactions=[r.text() for r in net], equations=format_network(net) if net else [],
               n_reactions=nr, n_params=int(p), n_data=n_data, n_runs=int(len(y)), chi2=chi2,
               chi2_per_point=chi2 / n_data, x_noise=float(np.sqrt(chi2 / n_data)),
               bic=float(chi2 + p * np.log(n_data)), model="plain" if kf is None else kf.model,
               miss=miss, unexplained=unexplained, not_made=not_made, fits_noise=fits_noise,
               seconds=float(seconds), a_spec=float(data["a_spec"]),
               predicted_gc=np.nan_to_num(yp, nan=0.0).round(6).tolist())
    out.update(_one_run(np.nan_to_num((yp - y) / (GC_REL * y + GC_ABS), nan=1e3)[:, :-1], p))
    out["rate_constants"], out["inhibition"] = [], None
    if kf is not None:
        out["scores"] = {m: float(v) for m, v in kf.scores.items()} or {"plain": out["bic"]}
        out["theta"], out["cov"] = kf.theta.tolist(), kf.cov.tolist()
        sd = kf.param_sd()
        for j, r in enumerate(net):
            a, b = kf.theta[2 * j], kf.theta[2 * j + 1]
            k = float(np.exp(a) * kf.a_spec)
            out["rate_constants"].append(dict(
                reaction=r.text(), rate_law=r.rate_law(j + 1), k_ref=k, k_ref_low=float(k * np.exp(-sd[2 * j])),
                k_ref_high=float(k * np.exp(sd[2 * j])), k_units=r.k_units(),
                Ea_kJ=float(b * R * T_REF / 1000.0), Ea_sd_kJ=float(sd[2 * j + 1] * R * T_REF / 1000.0)))
        if kf.model == "inhib":
            lnK, s_ = kf.theta[2 * nr], sd[2 * nr]
            out["inhibition"] = dict(K_C=float(np.exp(lnK)), K_C_low=float(np.exp(lnK - s_)), K_C_high=float(np.exp(lnK + s_)),
                                     units="1/bar at 300 C", adsorption_heat_kJ=float(-PRIOR_E * R * T_REF / 1000.0))
    else:
        out["scores"], out["theta"], out["cov"] = {"plain": out["bic"]}, [], []
    out["model_words"] = ("plain rate laws" if out["model"] == "plain" else
                          "rate laws with product C blocking catalyst sites, K_C = "
                          f"{out['inhibition']['K_C']:.3g} per bar at 300 C")
    out["verdict"] = _verdict(out, det)
    return out


def _verdict(f, det):
    """The fit in plain words.  When the network does not fit, one run carries most of the misfit
    and the other runs are explained within the allowance for model error, the run is named: a
    single bad reading looks like this, a wrong reaction does not."""
    txt = _verdict_core(f, det)
    if not f.get("fits_noise", f.get("fits")) and f.get("one_run") and f.get("rest_within_allowance"):
        txt += (f" (run {f['worst_run']} alone accounts for {100 * f['run_share']:.0f} % of the misfit: check its GC "
                "reading before blaming the reactions)")
    return txt


def _verdict_core(f, det):
    worst = max(f["miss"], key=lambda s: f["miss"][s]["x_noise"])
    if f["not_made"]:
        s = f["not_made"]
        txt = (f"cannot explain {' and '.join(s)}: no reaction in this network makes "
               f"{'it' if len(s) == 1 else 'them'} (the GC sees up to "
               + " and ".join(f"{det[x]:.1f} mol-% {x}" for x in s) + ")")
        rest = [x for x in f["unexplained"] if x not in s]
        return txt + (f"; it also misses {', '.join(rest)} by more than {MISS_LIMIT:g}x the GC noise" if rest else "")
    if f["unexplained"]:
        return ("cannot explain " + ", ".join(f["unexplained"]) + ": typical miss "
                + ", ".join(f"{f['miss'][s]['x_noise']:.1f}x" for s in f["unexplained"]) + " the GC noise ("
                + ", ".join(f"{f['miss'][s]['mol_pct']:.2f} mol-%" for s in f["unexplained"]) + ")")
    if f["fits_noise"]:
        return "fits the data within the GC noise"
    return (f"fits roughly: the misses are larger than GC noise alone would give (overall {f['x_noise']:.1f}x the "
            f"noise, worst for {worst} at {f['miss'][worst]['x_noise']:.1f}x)")


def misfit(network, data, yp, n_params=0):
    """How well model mole fractions explain the GC readings, in units of the GC's own noise.

    This is the yardstick of the whole module (the first half of what fit_network reports),
    offered on its own so that the advisor can ask it of ITS OWN kinetic fit after every run:
    "does my reaction network still explain what the GC prints?"

    network   the reactions the model uses (text, lines, Reaction objects or a ReactionSystem)
    data      the run log (load_data / make_data)
    yp        the model's mole fractions at the runs of the log, (n, 6) [A, B, C, D, E, I], or
              (n, 5) without E for the simple system
    n_params  number of fitted parameters behind yp (for the chi-square test)

    Returns a dict (JSON-friendly):
      x_noise      overall typical miss as a multiple of the GC noise (about 0.5 to 0.8 for a
                   network that is right, see the module docstring)
      chi2, n_data, n_runs
      miss         {species: {mol_pct, x_noise, bias_mol_pct}}: typical miss per species
      unexplained  species missed by more than MISS_LIMIT (2) times the GC noise, worst first
      not_made     species the GC sees but no reaction of the network makes
      detected     {species: largest mol-% seen}
      fits         True when the misses are no larger than GC noise alone would give (chi-square
                   test at the 1 % level) and no species is unexplained or not made
      verdict      the same in plain words

    A second, more forgiving yardstick is reported next to it.  No textbook model is exact (the
    virtual reactor has a reactant-adsorption term the advisor's model never has, and a catalyst
    that ages cannot be described before a set point has been repeated), so the kinetic fit
    weighs every reading with the GC noise PLUS an allowance for model error: 2 % of the reading
    + 0.2 mol-% (kinetics.GC_REL, GC_ABS; about 2 to 4 times the GC noise).  Misses within that
    allowance are what a RIGHT network leaves on a real reactor; only larger ones say that the
    network itself is wrong.
      x_allow           overall typical miss as a multiple of that allowance
      chi2_allow        the chi-square counted with the allowance (the kinetic fit's own weights)
      miss[s]['x_allow']  the same per species
      beyond            species missed by more than the allowance (x_allow > 1), worst first
      within_allowance  True when no species the GC sees is left unmade, none is missed by more
                        than the allowance, and the misses overall are within it (chi-square per
                        degree of freedom not above 1): the network is taken as adequate
    And whether the misfit sits in ONE run (a typing error or a bad GC injection looks like
    that; a wrong reaction shows wherever it matters), counted with the allowance:
      worst_run, run_share, one_run, rest_within_allowance   (see _one_run)
      worst_reading     {species, measured_mol_pct, predicted_mol_pct}: the reading of that run the
                        model misses most
    With yp of 5 columns (the simple system, whose GC has no E) the E column is left out of every
    number: an E the system cannot have is not a reading the network explains."""
    net = parse_network(network)
    y = data["gc"]
    yp = np.atleast_2d(np.asarray(yp, float))
    no_E = yp.shape[1] == 5
    if no_E:
        yp = np.insert(yp, 4, 0.0, axis=1)
    cols = [i for i, s in enumerate(data["species"][:-1]) if not (no_E and s == "E")]
    sig = gc_noise(y)
    z = np.nan_to_num((yp - y) / sig, nan=1e3)[:, cols]
    za = np.nan_to_num((yp - y) / (GC_REL * y + GC_ABS), nan=1e3)[:, cols]
    n_data = int(z.size)
    dof = max(n_data - int(n_params), 1)
    chi2, chi2a = float(np.sum(z ** 2)), float(np.sum(za ** 2))
    miss = {}
    for k, i in enumerate(cols):
        s = data["species"][i]
        d = np.nan_to_num(yp[:, i] - y[:, i], nan=1.0)
        miss[s] = dict(mol_pct=float(100 * np.sqrt(np.mean(d ** 2))), x_noise=float(np.sqrt(np.mean(z[:, k] ** 2))),
                       bias_mol_pct=float(100 * np.mean(d)), x_allow=float(np.sqrt(np.mean(za[:, k] ** 2))))
    det = detected_species(data)
    not_made = [s for s in det if s not in _made_by(net)]
    unexplained = sorted((s for s in miss if miss[s]["x_noise"] > MISS_LIMIT), key=lambda s: -miss[s]["x_noise"])
    beyond = sorted((s for s in miss if miss[s]["x_allow"] > 1.0), key=lambda s: -miss[s]["x_allow"])
    fits = bool(chi2 <= _chi2_dist.ppf(1 - FIT_LEVEL, dof)) and not unexplained and not not_made
    out = dict(x_noise=float(np.sqrt(chi2 / n_data)), chi2=chi2, n_data=n_data, n_runs=int(len(y)), miss=miss,
               unexplained=unexplained, not_made=not_made, detected=det, fits=fits,
               x_allow=float(np.sqrt(chi2a / n_data)), chi2_allow=chi2a, beyond=beyond,
               within_allowance=bool(not not_made and not beyond and chi2a / dof <= 1.0))
    out.update(_one_run(za, n_params))
    w, kw = out["worst_run"] - 1, int(np.argmax(np.abs(za[out["worst_run"] - 1])))
    out["worst_reading"] = dict(species=data["species"][cols[kw]], measured_mol_pct=float(100 * y[w, cols[kw]]),
                                predicted_mol_pct=float(100 * np.nan_to_num(yp[w, cols[kw]])))
    out["verdict"] = _verdict(dict(out, fits_noise=fits), det)
    return out


def _check_models(models):
    """The model classes asked for, as a tuple; a plain-word error for a class discovery does not use."""
    models = (models,) if isinstance(models, str) else tuple(models)
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ValueError(f"Unknown model class {', '.join(repr(m) for m in bad)}: reaction discovery uses 'plain' (plain "
                         "rate laws) and 'inhib' (product C blocking catalyst sites) only.")
    return models or ("plain",)


def _fit_net(net, data, models=MODELS, n_starts=4, seed=0, warm=None, name="network", system=None):
    t0 = time.time()
    models = _check_models(models)
    if not net:
        return _empty_fit(data, name)
    S = system if system is not None else build_system(net, name=name)
    kf = KineticFit(S, a_spec=data["a_spec"], seed=seed, models=tuple(models))
    if warm is not None:
        kf._warm = {"plain": np.asarray(warm, float)}
    kf.fit(data["phys"], data["gc"], hours=data["hours"], n_starts=n_starts)
    if kf.n_fits == 0:                       # every start failed: report the starting guess, honestly bad
        kf.theta, kf.model = kf.guess("plain"), "plain"
    return _score(net, kf, data, kf.predict_gc(data["phys"]), name, time.time() - t0)


def fit_network(system, data, models=MODELS, n_starts=4, seed=0, name=None):
    """Fit the rate constants and activation energies of ONE network to the data and say how
    well it explains them.

    system    reactions as text / lines / Reaction objects, or a ReactionSystem
    data      from load_data or make_data
    models    kinetic model classes allowed (see the module docstring); the data choose by BIC
    n_starts  starting points of the fit (the generic guess + random ones around it)

    Returns a dict (JSON-friendly):
      name, reactions [text], equations [numbered, with rate laws], n_reactions, n_params,
      n_data, n_runs, chi2 (against the GC noise), chi2_per_point, x_noise (overall typical miss
      as a multiple of the GC noise), bic, model ('plain' | 'inhib'), model_words, scores {model: bic},
      miss {species: {mol_pct, x_noise, bias_mol_pct}}   typical (root-mean-square) miss per species,
      unexplained [species missed by more than 2x the noise], not_made [species seen but made by
      no reaction], fits_noise (bool), verdict (plain words; it names the run when one run carries
      the misfit), worst_run, run_share, one_run, rest_within_allowance (see _one_run),
      rate_constants [{reaction, rate_law, k_ref, k_ref_low, k_ref_high, k_units, Ea_kJ, Ea_sd_kJ}]
          k_ref at 300 C per gram of catalyst (at the data's BET area), with its 1-sd range
      inhibition {K_C, K_C_low, K_C_high, units, adsorption_heat_kJ} or None,
      theta, cov (the fit in model units, for discriminating_run), predicted_gc, seconds, a_spec
    A ReactionSystem with other species than the complex system's (the built-in simple system has
    no E) is rebuilt from its reactions on the six species of the GC log, so its predictions line
    up with the log's columns."""
    if isinstance(system, ReactionSystem):
        net = parse_network(system)
        if list(system.species) != list(COMPLEX.species):
            return _fit_net(net, data, models, n_starts, seed, name=name or system.name)
        return _fit_net(net, data, models, n_starts, seed, name=name or system.name, system=system)
    net = parse_network(system)
    return _fit_net(net, data, models, n_starts, seed, name=name or "network")


# =================================================================================== ranking candidates
def _add_ranks(fits):
    fits = sorted(fits, key=lambda f: f["bic"])
    b0 = fits[0]["bic"]
    w = np.exp(-0.5 * np.minimum(np.array([f["bic"] - b0 for f in fits]), 1400.0))
    w = w / w.sum()
    for i, (f, wi) in enumerate(zip(fits, w), 1):
        f["rank"], f["delta_bic"], f["weight"] = i, float(f["bic"] - b0), float(wi)
        f["tied_with_best"] = bool(f["delta_bic"] < INDISTINGUISHABLE)
    return fits


def _difference(net_a, net_b):
    """In plain words, what distinguishes two networks."""
    a, b = parse_network(net_a), parse_network(net_b)
    only_a = [r for r in a if r.key() not in {q.key() for q in b}]
    only_b = [r for r in b if r.key() not in {q.key() for q in a}]
    if not only_a and not only_b:
        return "whether product C blocks catalyst sites"
    out = []
    for r in list(only_a):
        q = next((q for q in only_b if q.entry_key() == r.entry_key()), None)
        if q is None:
            continue
        only_a.remove(r)
        only_b.remove(q)
        if q.reversible != r.reversible:
            out.append(f"whether '{(q if r.reversible else r).equation()}' runs one way or both ways")
        elif q.direction_key() == r.direction_key() or r.reversible:
            out.append(f"the rate law of '{r.equation()}' ({r.rate_law().strip()}  or  {q.rate_law().strip()})")
        else:
            out.append(f"the direction of '{r.equation()}'")
    ta, tb = [f"'{r.text()}'" for r in only_a], [f"'{r.text()}'" for r in only_b]
    if ta and tb:
        out.append(f"whether {' and '.join(ta)} or {' and '.join(tb)} happens")
    elif ta or tb:
        out.append(f"whether {' and '.join(ta or tb)} happens at all")
    return "; ".join(out)


def rank_candidates(candidates, data, models=MODELS, n_starts=4, seed=0, progress=None):
    """Fit every candidate network and rank them by BIC.

    candidates  {name: network} or [(name, network)] or [network, ...]; a network is text, lines,
                Reaction objects or a ReactionSystem
    Returns a dict (JSON-friendly):
      ranking   the fit_network dicts, best first, each with rank, delta_bic (difference to the
                best), weight (evidence weight exp(-delta_bic/2), normalised) and tied_with_best
      best      name of the first, tied [names the data cannot tell apart from it], decided (bool)
      undecided [plain-word statements of what the data cannot decide]
      detected  {species: largest mol-% seen}, n_runs, n_settings (different settings among the
                runs: repeats count once), notes (about the log, from load_data), cautions (what so
                few or so similar runs cannot support), seconds, summary [plain-word lines, the
                notes and cautions last]"""
    t0 = time.time()
    if isinstance(candidates, dict):
        items = list(candidates.items())
    else:
        items = [c if isinstance(c, tuple) else (f"candidate {i}", c) for i, c in enumerate(candidates, 1)]
    if not items:
        raise ValueError("No candidate network was given: give at least one to rank.")
    fits = []
    for i, (name, net) in enumerate(items):
        if progress:
            progress(dict(stage="rank", message=f"fitting '{name}' ({i + 1} of {len(items)})", n_fits=i,
                          elapsed_s=time.time() - t0))
        fits.append(fit_network(net, data, models, n_starts, seed, name=name))
    return _ranking_result(fits, data, time.time() - t0)


def _few_runs_note(n, n_settings=None):
    """The caution for a log with fewer than FEW_RUNS different settings (n runs, n_settings of them different)."""
    head = (f"Caution: the log holds only {n} run{'s' if n != 1 else ''}." if n_settings is None or n_settings >= n else
            f"Caution: the log holds {n} runs but only {n_settings} different setting{'s' if n_settings != 1 else ''} "
            "(the others repeat them; a repeat checks the GC and the catalyst, it does not show how the reactions respond).")
    return (head + " With so few runs a wrong network can fit by chance, so take this as a first guess and confirm it "
            "with more runs at different conditions.")


def n_settings(data):
    """How many DIFFERENT settings the log holds (pressure to 0.1 bar, flows to 0.1 mL/min, temperature
    to 0.5 K, catalyst to 0.01 g): repeats of one setting count once."""
    return int(len(np.unique(np.round(data["phys"] / np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.01])), axis=0)))


def _cautions(data, n_params=0):
    """Plain-word cautions about what the log can support at all: few different settings, a single
    temperature, fewer different settings than fitted numbers.  The few-runs caution comes last."""
    n, k = len(data["gc"]), n_settings(data)
    out = []
    if k >= 2 and np.ptp(data["phys"][:, 4]) < 1.0:
        out.append("Every run was made at the same temperature, so the activation energies cannot be learned from these "
                   "runs: the values shown stay close to the starting guess (100 kJ/mol) and say nothing about the "
                   "chemistry. Runs at two or more temperatures are needed.")
    if FEW_RUNS <= k < n_params:
        out.append(f"The log holds {k} different settings, fewer than the {n_params} numbers fitted for the best network: "
                   "many networks can fit such data, so take the result as a first guess.")
    if k < FEW_RUNS:
        out.append(_few_runs_note(n, k))
    return out


def _ranking_result(fits, data, seconds):
    fits = _add_ranks(fits)
    best = fits[0]
    tied = [f["name"] for f in fits[1:] if f["tied_with_best"]]

    def same(f):                                     # the same reactions, rate laws and model class under another name
        return f["model"] == best["model"] and network_key(parse_network(f["reactions"])) == network_key(parse_network(best["reactions"]))

    def differs(f):
        d = _difference(best["reactions"], f["reactions"])
        return d + ("; and whether product C blocks catalyst sites" if f["model"] != best["model"] and "blocks catalyst sites" not in d else "")

    undecided = [f"'{best['name']}' and '{f['name']}' are the same network (the same reactions and rate laws), so nothing can tell them apart."
                 if same(f) else
                 f"'{best['name']}' and '{f['name']}' explain the data equally well (BIC difference {f['delta_bic']:.1f}, "
                 f"below {INDISTINGUISHABLE:g}): the data cannot decide {differs(f)}."
                 for f in fits[1:] if f["tied_with_best"]]
    summary = []
    for f in fits:
        rel = (("best" if best["fits_noise"] else "the least bad of the candidates") if f["rank"] == 1 else
               f"cannot be told apart from the best (BIC difference {f['delta_bic']:.1f})" if f["tied_with_best"] else
               f"ruled out against the best (BIC difference {f['delta_bic']:.0f})")
        summary.append(f"{f['rank']}. {f['name']}: {f['verdict']}; {rel}.")
    if not best["fits_noise"]:
        summary.append("Note: even the best candidate does not fit within the GC noise, so something is happening "
                       "that none of the candidates describes.")
    notes, cautions = list(data.get("notes", [])), _cautions(data, best["n_params"])
    summary += notes + cautions
    return dict(ranking=fits, best=best["name"], tied=tied, decided=not tied, undecided=undecided,
                detected=detected_species(data), n_runs=int(len(data["gc"])), n_settings=n_settings(data),
                notes=notes, cautions=cautions, seconds=float(seconds), summary=summary)


# =================================================================================== the reaction library
def _equilibrium_reach(r, table):
    """Smallest and largest ln of the mole-fraction equilibrium constant of reaction r over the
    operating box (K_x = K(T) P^-dn, dn = change in the number of molecules)."""
    dH, dS = r.thermo(table)
    dn = sum(r.products.values()) - sum(r.reactants.values())
    T = np.linspace(LOW[4], HIGH[4], 9)[:, None]
    P = np.array([LOW[0], HIGH[0]])[None, :]
    lnKx = -dH / (R * T) + dS / R - dn * np.log(P)
    return float(lnKx.min()), float(lnKx.max())


def reaction_library(species=None, table=None, max_reactants=3, max_products=2, thermo_prune=True):
    """Every atom-balanced reaction among the non-inert species with small coefficients,
    generated by ENUMERATION: all pairs of 'sides' (1 to 3 molecules) that hold the same atoms
    and share no species, with at most `max_reactants` molecules reacting and `max_products`
    formed in at least one direction.  A reaction and its reverse are ONE entry with variants:

        'forward'     one-way as written          'reverse'     one-way, the other way round
        'reversible'  both ways, equilibrium constant from the species table

    A one-way variant needs its own direction to respect the molecule limits.  With
    thermo_prune, thermodynamics removes what can never be seen in the operating box: if one
    direction is complete everywhere (K_x > 100 at every temperature and pressure of the box)
    only that one-way variant is kept (its reverse cannot run, and 'reversible' would predict
    exactly the same).

    Returns a list of dicts: key, equation (forward, as text), dH_kJ, dS_J_K,
    variants [{kind, text}]; texts are in the reaction text format."""
    table = SPECIES_TABLE if table is None else table
    species = [s for s in (species or table) if table[s]["role"] != "inert"]
    sides = {}
    for n in range(1, max(max_reactants, max_products) + 1):
        for combo in combinations_with_replacement(species, n):
            side = {s: combo.count(s) for s in dict.fromkeys(combo)}
            atoms = tuple(sorted(atom_count(side, table).items()))
            sides.setdefault(atoms, []).append(side)
    out, seen = [], set()
    for group in sides.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if set(a) & set(b):
                    continue
                if math.gcd(*a.values(), *b.values()) > 1:          # a multiple of a simpler reaction
                    continue
                na, nb = sum(a.values()), sum(b.values())
                ok_ab = na <= max_reactants and nb <= max_products    # a -> b allowed as a one-way step
                ok_ba = nb <= max_reactants and na <= max_products
                if not (ok_ab or ok_ba):
                    continue
                # forward = the direction that joins molecules (or, for equal counts, releases heat)
                r = Reaction(a, b, False, table=table)
                swap = (not ok_ab) or (ok_ba and (nb > na or (nb == na and r.thermo(table)[0] > 0)))
                fwd_side, rev_side, ok_f, ok_r = (b, a, ok_ba, ok_ab) if swap else (a, b, ok_ab, ok_ba)
                fwd = Reaction(fwd_side, rev_side, False, table=table)
                if fwd.entry_key() in seen:
                    continue
                seen.add(fwd.entry_key())
                lo, hi = _equilibrium_reach(fwd, table)
                if thermo_prune and hi < -np.log(K_COMPLETE):         # only the other direction can run: list it that way round
                    fwd_side, rev_side, ok_f, ok_r = rev_side, fwd_side, ok_r, ok_f
                    fwd = Reaction(fwd_side, rev_side, False, table=table)
                    lo, hi = -hi, -lo
                variants = []
                complete_f = thermo_prune and lo > np.log(K_COMPLETE)
                complete_r = thermo_prune and hi < -np.log(K_COMPLETE)
                if ok_f and not complete_r:
                    variants.append(dict(kind="forward", text=fwd.text()))
                if ok_r and not complete_f:
                    variants.append(dict(kind="reverse", text=fwd.reversed().text()))
                if not (complete_f or complete_r):
                    variants.append(dict(kind="reversible", text=Reaction(fwd_side, rev_side, True, table=table).text()))
                dH, dS = fwd.thermo(table)
                if variants:
                    out.append(dict(key=fwd.entry_key(), equation=fwd.equation(), dH_kJ=dH / 1000.0, dS_J_K=dS,
                                    variants=variants))
    return sorted(out, key=lambda e: (len(e["equation"]), e["equation"]))


def _library_reactions(library):
    lib = reaction_library() if library is None else library
    out = []
    for e in lib:
        if isinstance(e, dict):
            out += [parse_reaction(v["text"]) for v in e["variants"]]
        else:
            out += parse_network([e])
    return out


# =================================================================================== the automatic search
def _reachable(net, fed):
    """True if every reaction can fire starting from the fed species (its reactants are fed or
    made by another reaction of the network)."""
    have, todo = set(fed), list(net)
    while todo:
        fire = [r for r in todo if set(r.reactants) <= have or (r.reversible and set(r.products) <= have)]
        if not fire:
            return False
        for r in fire:
            have |= set(r.products) | set(r.reactants)
            todo.remove(r)
    return True


def _warm_theta(net, old_net, old_theta):
    """Starting point for a changed network: reactions kept from the old network keep their
    fitted constants, new ones start from the generic guess."""
    gen = generic_theta(1)
    have = {r.direction_key(): old_theta[2 * j:2 * j + 2] for j, r in enumerate(old_net)} if len(old_theta) else {}
    return np.concatenate([np.asarray(have.get(r.direction_key(), gen), float) for r in net])


def _order_variants(r, choices=ORDER_CHOICES):
    """The same reaction with ONE reactant order changed to another value of `choices`.
    A reversible reaction cannot have an order below the coefficient (its reverse rate would
    become infinite), so for such an order the ONE-WAY form of the reaction is offered: without
    it a reaction the search first took as reversible could never be found to be, say, zero
    order in a reactant."""
    out = []
    for s, c in r.reactants.items():
        for o in choices:
            if o == r.orders.get(s, 0):
                continue
            try:
                out.append(Reaction(r.reactants, r.products, r.reversible and o >= c, dict(r.orders, **{s: float(o)})))
            except ReactionError:
                pass
    return out


N_SCREEN = 4        # iterations of the quick fit that screens the possible changes of a search step
MAX_RECHECKS = 2    # how often the reactions are checked again after the order check changed a rate law
ORDERS_IN_SEARCH = True   # a change of one reaction order competes with adding / removing a reaction at every step
WIDE_RESTARTS = True      # from its second new start on, the search also tries every reaction it may add with one order changed


def _batch_lm(nets, data, starts, n_iter=N_SCREEN, tol=1e-4, inhib=None):
    """Fit SEVERAL networks at once with plain rate laws (damped Gauss-Newton, i.e.
    Levenberg-Marquardt, on the objective of kinetics.KineticFit: GC residuals with the fit
    weights plus the wide prior around the generic guess).

    The search has to compare dozens of networks per step.  Integrating the bed for one
    network at a time spends most of its time on bookkeeping, so all networks of a step are
    integrated together: every row (one run of one network, or one of its perturbed copies for
    the derivatives) carries the orders and stoichiometry of its own network (_bed), and each
    iteration needs a single integration for all networks.  A row gives exactly the numbers it
    would give alone (checked in test_discovery.py).

    nets, starts   networks (lists of Reaction) and their starting parameter vectors
    inhib          None (plain rate laws), or one entry per network: None, or K_C at the
                   temperature of every run (n,), 1/bar: product C blocking catalyst sites at a
                   GIVEN strength (every rate / (1 + K_C P_C)^m_j, as kinetics.KineticFit's
                   'inhib' model), not fitted here.  Used by one_step_revision to compare changes
                   on the footing of a caller whose own fit found site blocking.
    Returns one dict per network: theta, r (weighted residuals), J, yp (model mole fractions),
    cost, done (True if converged within n_iter)."""
    phys, y = data["phys"], data["gc"]
    n = len(phys)
    systems = [build_system(net, unique=False) for net in nets]
    nq, ns = max(S.nr for S in systems), systems[0].ns
    FWD, REV, ZERO = (np.zeros((len(nets), nq, ns)) for _ in range(3))
    NU = np.zeros((len(nets), ns, nq))
    P, VA, VB, VI, T, W = phys.T
    KEQ = np.full((len(nets), n, nq), np.inf)                    # ln K_eq; +inf = one-way (or padding)
    for c, S in enumerate(systems):
        FWD[c, :S.nr], REV[c, :S.nr], NU[c, :, :S.nr], KEQ[c, :, :S.nr] = S.fwd, S.rev, S.nu, S.log_keq(T)
        for j, i in S._zero:
            ZERO[c, j, i] = 1.0
    F0 = systems[0].inlet_flows(VA, VB, VI)
    f0, sc, lnP, tfac = F0 / F0.sum(1, keepdims=True), data["a_spec"] * W / F0.sum(1), np.log(P), T_REF / T - 1.0
    sig = GC_REL * y[:, :-1] + GC_ABS
    eps, gen = 1e-4, generic_theta(1)
    KPM = None                                                   # K_C x P for every (network, run), only if asked for
    if inhib is not None and any(k is not None for k in inhib):
        KPM = np.zeros((len(nets), n))
        for c, k in enumerate(inhib):
            if k is not None:
                KPM[c] = np.broadcast_to(np.asarray(k, float), (n,)) * P

    def evaluate(items):
        """items [(network index, theta)] -> [(residuals, Jacobian, model mole fractions)]"""
        ci, ri, grp, lnk = [], [], [], []
        for g, (c, th) in enumerate(items):
            p = len(th)
            TH = np.repeat(th[None, :], p + 1, 0)
            TH[1:] += eps * np.eye(p)
            lk = np.full((p + 1, n, nq), -np.inf)                # -inf = no such reaction in this network
            lk[:, :, :p // 2] = TH[:, None, 0::2] - TH[:, None, 1::2] * tfac[None, :, None]
            lnk.append(lk.reshape(-1, nq))
            idx = np.tile(np.arange(n), p + 1)
            ri.append(idx)
            ci.append(np.full(len(idx), c))
            grp.append(g * n + idx)                              # an experiment and its perturbed copies share steps
        ci, ri = np.concatenate(ci), np.concatenate(ri)
        keq = KEQ[ci, ri]
        with np.errstate(invalid="ignore"):
            f = _bed(FWD[ci], REV[ci], NU[ci], ZERO[ci] if ZERO[ci].any() else None, np.vstack(lnk),
                     keq if np.isfinite(keq).any() else None, lnP[ri], sc[ri], f0[ri],
                     KP=None if KPM is None else KPM[ci, ri][:, None], groups=np.concatenate(grp))
            f = np.where(np.isnan(f), np.nan, np.maximum(f, 0.0))
        Y = f / f.sum(1, keepdims=True)
        out, o = [], 0
        for c, th in items:
            p = len(th)
            Yc = Y[o:o + (p + 1) * n].reshape(p + 1, n, -1)
            o += (p + 1) * n
            r0 = np.nan_to_num(((Yc[0, :, :-1] - y[:, :-1]) / sig).ravel(), nan=1e3)
            Jd = np.nan_to_num(((Yc[1:, :, :-1] - Yc[0, :, :-1][None]) / eps / sig[None]).reshape(p, -1).T, nan=0.0)
            psd = np.tile([PRIOR_SD_A, PRIOR_SD_B], p // 2)
            out.append((np.r_[r0, (th - np.tile(gen, p // 2)) / psd], np.vstack([Jd, np.diag(1.0 / psd)]), Yc[0]))
        return out

    starts = [np.asarray(t, float) for t in starts]
    state = [dict(theta=th.copy(), r=r, J=J, yp=yp, cost=float(r @ r), lam=1e-2, done=False)
             for th, (r, J, yp) in zip(starts, evaluate(list(enumerate(starts))))]
    for _ in range(n_iter):
        trials = []
        for c, st in enumerate(state):
            if st["done"]:
                continue
            A, g = st["J"].T @ st["J"], st["J"].T @ st["r"]
            try:
                d = np.linalg.solve(A + st["lam"] * np.diag(np.diag(A)), -g)
            except np.linalg.LinAlgError:
                d = -np.linalg.pinv(A + st["lam"] * np.diag(np.diag(A))) @ g
            d[0::2], d[1::2] = np.clip(d[0::2], -4.0, 4.0), np.clip(d[1::2], -12.0, 12.0)
            trials.append((c, st["theta"] + d))
        if not trials:
            break
        for (c, th), (r, J, yp) in zip(trials, evaluate(trials)):
            st, cost = state[c], float(r @ r)
            if cost < st["cost"]:
                st["done"] = st["cost"] - cost < tol * st["cost"]
                st.update(theta=th, r=r, J=J, yp=yp, cost=cost, lam=max(st["lam"] / 5.0, 1e-9))
            else:
                st["lam"] *= 6.0
                st["done"] = st["lam"] > 1e7
    return state


def _state_fit(net, st, data, seconds=0.0):
    """A _batch_lm result -> the dict fit_network returns (plain rate laws)."""
    kf = KineticFit(build_system(net), a_spec=data["a_spec"], models=("plain",))
    p = len(st["theta"])
    n_data = st["r"].size - p
    kf.theta, kf.model = st["theta"], "plain"
    kf.chi2_dof = float(np.sum(st["r"][:n_data] ** 2)) / max(n_data - p, 1)
    H = st["J"].T @ st["J"]
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    kf.cov = 0.5 * (cov + cov.T) * max(1.0, kf.chi2_dof)
    return _score(net, kf, data, st["yp"], "network", seconds)


def discover(data, library=None, start=None, max_reactions=4, models=MODELS, progress=None, seed=0,
             n_polish=3, n_final=4, max_seconds=None, order_choices=ORDER_CHOICES, n_restarts=3):
    """Search for the reaction network that explains the data best (lowest BIC).

    data           from load_data / make_data
    library        reaction_library() by default (enumerated from the atom balance)
    start          None = start from no reactions at all; or a network (text, lines, Reaction
                   objects) to REPAIR: the search then adds, removes and replaces from there
    max_reactions  largest network considered
    models         model classes for the final re-fit of the best networks (the search itself
                   uses plain rate laws)
    progress       optional callback, called with a dict {stage, message, n_fits, elapsed_s, best_bic}
    max_seconds    optional time limit for the structure search (the result says if it was hit)
    n_restarts     how many times the search may start again from another first reaction when
                   what it found does not fit within the GC noise (see below); 0 = never

    The search:  each step tries every allowed single change (ADD a library reaction whose
    reactants are fed or already made, REMOVE a reaction, once every product seen is made by
    some reaction also CHANGE ONE REACTION ORDER, and, when none of these helps, REPLACE a
    reaction by another variant of itself or by another reaction making the same product),
    fits each changed network and takes the best if it lowers BIC by more than MIN_GAIN.
    Additions are only tried while some species is still missed by more than the GC noise
    allows, and only reactions touching such a species are tried; while a product the GC sees
    is made by no reaction yet, only reactions that make such a product are tried.  All changes
    of a step are first compared by a quick fit in one batch (N_SCREEN iterations from the
    current fit) and the best `n_polish` are fitted to convergence.  When the reactions are
    settled, each reactant order is tried at the other values of order_choices (0, 1, 2).
    If an order changes, the reactions are checked once more with the new rate laws (at most
    MAX_RECHECKS times): the reactions were chosen assuming mass-action orders, and when the
    true rate is, say, zero order in a reactant, the search first adds an extra reaction to
    make up for it, which must be removed again once the order is right.
    Final check: the best `n_final` networks are re-fitted with `models` (all of them, so that
    the BIC differences reported compare like with like; when the best network still misses by
    more than the GC noise, it is re-fitted from several starting points).  If site blocking
    turns out to be supported, reactions are dropped one at a time
    while that does not raise BIC: the plain search may have added a reaction only to imitate
    the site blocking.  Networks with the same BIC (to 0.5) are listed with the one that has
    fewer two-way reactions first.
    Starting again: the search changes ONE reaction at a time, so a wrong early choice can lead
    it into a dead end (seen with the 8 runs of a short step-limited campaign, where one odd
    reaction stood in for two real ones).  It notices this by the result: if the best network does not
    fit within the GC noise, it searches again from the next-best first reaction (a repair
    first searches again from no reactions), up to n_restarts times, and returns the best of
    everything fitted.  The first new start is searched like the first pass.  From the second
    on (WIDE_RESTARTS) every reaction that may be added is also tried with one of its orders
    changed (0, 1 or 2 for one reactant): two failed passes suggest that a rate does not
    follow its equation, which the passes before only test once the reactions are settled.
    Such a pass fits several times more networks per step.  A result that fits is NOT
    searched again, so most of the examples take one pass; data that no network of the
    library can fit cost up to n_restarts more passes and several minutes (pass max_seconds
    or a smaller n_restarts to limit that).

    Returns a dict (JSON-friendly):
      best          the fit_network dict of the best network (+ rank, delta_bic, weight)
      runners_up    the next networks, same form, with delta_bic and what distinguishes each
                    from the best ('differs_by')
      tied          names of runners-up the data cannot tell apart from the best
      decided       False when there is such a tie
      undecided     plain-word statements of what the data cannot decide
      path          the search, step by step, in plain words
      complete      True if every species the GC sees is made by the best network
      n_fits, seconds, library_size, detected, stopped_early (bool), restarts (how often the
      search started again), n_runs, n_settings, notes, cautions (as rank_candidates),
      summary [plain-word lines]
    max_seconds bounds the whole search: when it is reached the final check (re-fits with site
    blocking, the pruning after it) is cut short and the path says so; one batch of fits that was
    already running can take the total a few seconds past the limit."""
    t0 = time.time()
    models = _check_models(models)
    lib = [r for r in _library_reactions(library)]
    fed = fed_species(data)
    det = detected_species(data)
    visible = set(fed) | set(det)
    lib = [r for r in lib if set(r.products) <= visible or (r.reversible and set(r.reactants) | set(r.products) <= visible)]
    cache, count = {}, [0]
    path, stopped = [], False

    def say(stage, message, best=None):
        if progress:
            progress(dict(stage=stage, message=message, n_fits=count[0], elapsed_s=time.time() - t0,
                          best_bic=None if best is None else best["bic"]))

    rng = np.random.default_rng(seed)
    polished = set()

    def keep(net, f):
        key = network_key(net)
        if key not in cache or f["bic"] < cache[key]["bic"]:
            cache[key] = f
        return cache[key]

    def out_of_time():
        return max_seconds is not None and time.time() - t0 > max_seconds

    def best_of(moves, cur, label, random_start=False):
        """Quick-fit every changed network in ONE batch (starting from the current fit), then
        fit the best few to convergence.  Networks fitted earlier in the search are not fitted
        again.  Returns [(fit, what, network)], best first."""
        if cur is not None and cur["n_reactions"]:
            old, th_old = parse_network(cur["reactions"]), np.asarray(cur["theta"], float)[:2 * cur["n_reactions"]]
        else:
            old, th_old = [], np.zeros(0)
        nets = [net for _, net in moves]
        keys = [network_key(net) for net in nets]
        fresh = [i for i, k in enumerate(keys) if k not in cache]
        if fresh:
            t = time.time()
            states = _batch_lm([nets[i] for i in fresh], data, [_warm_theta(nets[i], old, th_old) for i in fresh], N_SCREEN)
            count[0] += len(fresh)
            for i, st in zip(fresh, states):
                keep(nets[i], _state_fit(nets[i], st, data, (time.time() - t) / len(fresh)))
        say("search", f"{label}: compared {len(nets)} possible changes", cur)
        order = sorted(range(len(nets)), key=lambda i: cache[keys[i]]["bic"])
        # only changes that could still beat the current network are worth fitting to convergence
        limit = np.inf if cur is None else 2.0 * cur["bic"] + 50.0
        top = [i for i in order[:n_polish] if keys[i] not in polished and cache[keys[i]]["bic"] < limit]
        if top:
            pn, ps = [], []
            for i in top:
                th = np.asarray(cache[keys[i]]["theta"], float)
                pn.append(nets[i])
                ps.append(th)
                if random_start:
                    pn.append(nets[i])
                    ps.append(generic_theta(len(th) // 2)
                              + 0.7 * np.tile([PRIOR_SD_A, PRIOR_SD_B], len(th) // 2) * rng.standard_normal(len(th)))
            t = time.time()
            pol = _batch_lm(pn, data, ps, 20, tol=1e-6)
            count[0] += len(pn)
            for net, st in zip(pn, pol):
                keep(net, _state_fit(net, st, data, (time.time() - t) / len(pn)))
            polished.update(keys[i] for i in top)
        return sorted(((cache[keys[i]], moves[i][0], nets[i]) for i in range(len(nets))), key=lambda z: z[0]["bic"])

    firsts = set()                             # first reactions from which a search from nothing has already grown

    def grow(net, cur, wide=False):
        """One greedy search from the network `net` (fitted: cur): the structure, then the orders,
        and, whenever an order changed, the structure once more (see tune_orders).
        wide: every reaction that may be added is also tried with one of its orders changed (used
        when the search starts again: see WIDE_RESTARTS).
        Returns the network it ends at, its fit, and True if the time limit stopped it."""
        step, halted, rechecks, changed = 0, False, 0, True
        while True:
            step += 1
            if out_of_time():
                halted = True
                break
            have = set(fed) | _made_by(net)
            entries = {r.entry_key() for r in net}
            poor = {s for s, m in cur["miss"].items() if m["x_noise"] > 1.3} | set(cur["not_made"])
            moves = []
            orphan = set(cur["not_made"])          # products the GC sees that no reaction makes yet: explain those first
            if len(net) < max_reactions and poor:
                for r in lib:
                    makes = set(r.products) | (set(r.reactants) if r.reversible else set())
                    touches = (makes & orphan) if orphan else ((set(r.reactants) | set(r.products)) & poor)
                    starts_ok = set(r.reactants) <= have or (r.reversible and set(r.products) <= have)
                    if r.entry_key() not in entries and starts_ok and touches:
                        moves.append((f"add '{r.text()}'", net + [r]))
                        if wide:
                            moves += [(f"add '{q.text()}'", net + [q]) for q in _order_variants(r, order_choices)]
            n_add = len(moves)
            for j, r in enumerate(net):
                rest = net[:j] + net[j + 1:]
                if rest and _reachable(rest, fed):
                    moves.append((f"remove '{r.text()}'", rest))
            scored = best_of(moves, cur, f"step {step}") if moves else []
            tried = (f"tried {n_add} addition{'s' if n_add != 1 else ''}" + (" (each reaction also with other orders)" if wide and n_add else "")
                     + f" and {len(moves) - n_add} removal{'s' if len(moves) - n_add != 1 else ''}")
            if len(net) >= max_reactions:
                tried += f" (no additions: {max_reactions} reaction{'s' if max_reactions != 1 else ''} is the largest network allowed)"
            elif net and not poor:
                tried += " (no additions: every species is already explained within the GC noise)"
            accept = None
            if scored:
                f, what, new = scored[0]
                gain = cur["bic"] - f["bic"]
                if gain > MIN_GAIN or (what.startswith("remove") and gain >= 0):
                    accept = (f, what, new)
            # Once every product the GC sees is made by some reaction, a change of ONE reaction order
            # competes with the additions and removals (unless one of those already gives a network that
            # fits).  Without this a rate that is, say, zero order in a reactant is patched with extra
            # reactions before the order is ever tried, and the search can end in a dead end.
            if ORDERS_IN_SEARCH and net and not orphan and not (accept and accept[0]["fits_noise"]) and not out_of_time():
                var = [(f"change the rate law of '{r.equation()}' to {q.rate_law().strip()} (was {r.rate_law().strip()})",
                        net[:j] + [q] + net[j + 1:]) for j, r in enumerate(net) for q in _order_variants(r, order_choices)]
                if var:
                    oscored = best_of(var, cur, f"step {step} (orders)")
                    tried += f" and {len(var)} other reaction order{'s' if len(var) != 1 else ''}"
                    if cur["bic"] - oscored[0][0]["bic"] > MIN_GAIN and (accept is None or oscored[0][0]["bic"] < accept[0]["bic"]):
                        accept = oscored[0]
            if accept is None and net and not out_of_time():
                # replacements: first another variant of the same reaction (other direction, one way /
                # both ways), then another route to the same product
                same, other = [], []
                for j, r in enumerate(net):
                    others = entries - {r.entry_key()}
                    made = set(r.products) & set(det)
                    for q in lib:
                        twin = q.entry_key() == r.entry_key()
                        if q.key() == r.key() or q.entry_key() in others or not (twin or set(q.products) & made):
                            continue
                        new = net[:j] + [q] + net[j + 1:]
                        if _reachable(new, fed):
                            (same if twin else other).append((f"replace '{r.text()}' by '{q.text()}'", new))
                n_rep = 0
                for rep in (same, other):
                    if not rep or out_of_time():
                        continue
                    n_rep += len(rep)
                    rscored = best_of(rep, cur, f"step {step} (replacements)")
                    if cur["bic"] - rscored[0][0]["bic"] > MIN_GAIN:
                        accept = rscored[0]
                        break
                tried += f", then {n_rep} replacement{'s' if n_rep != 1 else ''}"
            if accept is None:
                path.append(f"Step {step}: {tried}; none lowers BIC by more than {MIN_GAIN:g}. The reactions are settled.")
                if not changed:                    # a re-check after new orders that changed no reaction: the orders stand
                    break
                net, cur, new_orders = tune_orders(net, cur)
                if not new_orders or rechecks >= MAX_RECHECKS or out_of_time():
                    break
                # The reactions were chosen with the OLD rate laws.  A reaction that was only added to
                # make up for a wrong order (seen when the true rate is zero order in a reactant) is no
                # longer needed once the order is right, so the reactions are checked again.
                rechecks, changed = rechecks + 1, False
                path.append("A rate law changed, so the reactions are checked once more: a reaction added only to make "
                            "up for the old rate law may no longer be needed.")
                continue
            f, what, new = accept
            path.append(f"Step {step}: {tried}. Best: {what} (BIC {cur['bic']:.0f} -> {f['bic']:.0f}); now {f['verdict']}.")
            if not net:
                firsts.add(new[0].entry_key())
            net, cur, changed = new, f, True
            say("search", path[-1], cur)
        if halted or out_of_time():
            halted = True
            path.append(f"The search was stopped by the time limit ({max_seconds:g} s); the result is the best network found so far.")
        return net, cur, halted

    def tune_orders(net, cur):
        """The reaction orders of a settled network: each reactant order is tried at the other
        values of order_choices, the best change is kept, up to three times.
        Returns the network, its fit, and True if any order changed."""
        moved = False
        for rnd in range(3):
            if not net or out_of_time():
                break
            var = []
            for j, r in enumerate(net):
                for q in _order_variants(r, order_choices):
                    var.append((f"'{r.equation()}': {q.rate_law().strip()} instead of {r.rate_law().strip()}", net[:j] + [q] + net[j + 1:]))
            scored = best_of(var, cur, "order check") if var else []
            if scored and cur["bic"] - scored[0][0]["bic"] > MIN_GAIN:
                f, what, new = scored[0]
                path.append(f"Order check: tried {len(var)} other order{'s' if len(var) != 1 else ''}. Better: {what} (BIC {cur['bic']:.0f} -> {f['bic']:.0f}).")
                net, cur, moved = new, f, True
            else:
                path.append(f"Order check: tried {len(var)} other order{'s' if len(var) != 1 else ''} (each reactant at {', '.join(map(str, order_choices))}); "
                            "none is better than the present rate laws." if var else "Order check: nothing to try.")
                break
        return net, cur, moved

    def top_plain():
        return sorted((f for f in cache.values() if f["n_reactions"] > 0), key=lambda f: f["bic"])[:n_final]

    refits = {}                                # networks re-fitted with `models` in a final check

    def key_of(f):
        return network_key(parse_network(f["reactions"]))

    def finish(cur):
        """Final check of everything fitted so far: converge the runners-up; if the best does not
        fit within the GC noise, allow the other model classes.  Returns the fits, best first.
        When the time limit has been reached the final check is skipped (and the path says so), so
        that max_seconds bounds the whole search, not only its structure part."""
        say("final", "final check of the best networks", cur)
        if out_of_time():
            path.append("The time limit was reached before the final check: the networks are listed as the search fitted "
                        "them (plain rate laws; the runners-up only roughly; site blocking was not tried).")
            final = sorted([dict(f) for f in top_plain()] or [dict(cur)], key=lambda f: f["bic"])
            return final
        loose = [parse_network(f["reactions"]) for f in top_plain()]
        loose = [n_ for n_ in loose if network_key(n_) not in polished]
        if loose:                                        # runners-up seen only in a quick fit: fit them to convergence
            for n_, st in zip(loose, _batch_lm(loose, data, [np.asarray(cache[network_key(n_)]["theta"], float) for n_ in loose], 20, tol=1e-6)):
                keep(n_, _state_fit(n_, st, data))
            polished.update(network_key(n_) for n_ in loose)
            count[0] += len(loose)
        plain = top_plain()
        # Site blocking and extra starting points are only worth trying when plain rate laws leave
        # misses larger than the GC noise; and only for networks within reach of the best one.
        refit = bool(plain) and not plain[0]["fits_noise"]
        # When the best network already fits, it is still fitted once more with the other model classes,
        # so that the search reports the same model class and the same constants for a network as
        # rank_candidates does (both keep site blocking only when it lowers BIC).  The runners-up get
        # exactly the same treatment: the BIC differences reported must compare like with like (a
        # best network allowed site blocking against runners-up that were not would look more
        # clearly ahead than it is, and a tie could go unreported).
        for i, f in enumerate(plain if refit or len(models) > 1 else []):
            if key_of(f) in refits or (i and f["bic"] > 10.0 * plain[0]["bic"] + 100.0):
                continue
            if i and out_of_time():                      # the best one is always re-fitted; the rest only in time
                path.append("The time limit was reached during the final check: the next best networks were not "
                            "re-fitted with site blocking allowed.")
                break
            say("final", f"re-fitting the best networks, allowing site blocking ({i + 1} of {len(plain)})" if refit else
                "does product C blocking catalyst sites improve the best network?" if i == 0 else
                f"the same check for the next best networks ({i} of {len(plain) - 1})", cur)
            refits[key_of(f)] = _fit_net(parse_network(f["reactions"]), data, tuple(models), 3 if i == 0 and refit else 1, seed,
                                         warm=np.asarray(f["theta"], float)[:2 * f["n_reactions"]])
            count[0] += 1
        final = [dict(refits[key_of(f)]) if key_of(f) in refits and refits[key_of(f)]["bic"] <= f["bic"] else dict(f)
                 for f in plain] or [dict(cur)]
        final += [dict(g) for k, g in refits.items() if k not in {key_of(f) for f in final}]
        final.sort(key=lambda f: f["bic"])
        if not refit and final and final[0]["model"] != "plain" and key_of(final[0]) != key_of(plain[0]):
            path.append("Final check: the best network found with plain rate laws fits within the GC noise, but when product C "
                        "is allowed to block catalyst sites (one more fitted number; every network listed was given the same "
                        f"chance) another network explains the data better still (BIC {plain[0]['bic']:.0f} -> {final[0]['bic']:.0f}); "
                        "it is listed first.")
        elif not refit and final and final[0]["model"] != "plain":
            path.append("Final check: the best network fits within the GC noise with plain rate laws, and better still when "
                        f"product C is allowed to block catalyst sites (BIC {plain[0]['bic']:.0f} -> {final[0]['bic']:.0f}), "
                        "so site blocking is kept. The next best networks were given the same chance.")
        elif not refit:
            path.append("Final check: the best network already fits within the GC noise with plain rate laws"
                        + ("; letting product C block catalyst sites (one more fitted number) does not improve on that."
                           if len(models) > 1 and plain else "."))
        elif len(models) > 1:
            path.append("Final check: plain rate laws leave misses larger than the GC noise, so the best networks were "
                        "re-fitted allowing product C to block catalyst sites (the best one from several starting points). "
                        + ("That does not help." if final[0]["model"] == "plain" else
                           f"That lowers BIC from {plain[0]['bic']:.0f} to {final[0]['bic']:.0f}"
                           + (": the best network now fits within the GC noise, so the data support site blocking."
                              if final[0]["fits_noise"] else ", but even so the best network does not fit within the GC noise.")))
        while len(models) > 1 and final[0]["model"] != "plain" and final[0]["n_reactions"] > 1 and not out_of_time():
            top, tnet, better = final[0], parse_network(final[0]["reactions"]), None
            rests = [tnet[:j] + tnet[j + 1:] for j in range(len(tnet))]
            fresh = [n_ for n_ in rests if network_key(n_) not in cache and _reachable(n_, fed)]
            if fresh:                                    # a quick plain fit shows which removals are hopeless
                th_top = np.asarray(top["theta"], float)[:2 * len(tnet)]
                for n_, st in zip(fresh, _batch_lm(fresh, data, [_warm_theta(n_, tnet, th_top) for n_ in fresh], N_SCREEN)):
                    keep(n_, _state_fit(n_, st, data))
                count[0] += len(fresh)
            for j, r in enumerate(tnet):
                rest = rests[j]
                seen = cache.get(network_key(rest))
                if not _reachable(rest, fed) or (seen is not None and seen["x_noise"] > 10.0):
                    continue                             # misses by 10x the noise with plain rate laws: site blocking cannot mend that
                say("final", f"with site blocking allowed: is '{r.text()}' still needed?", top)
                g = next((f for f in final if key_of(f) == network_key(rest)), None)
                if g is None:
                    g = _fit_net(rest, data, tuple(models), 1, seed, warm=_warm_theta(rest, tnet, np.asarray(top["theta"], float)[:2 * len(tnet)]))
                    count[0] += 1
                    refits[network_key(rest)] = g
                    final.append(dict(g))
                if g["bic"] <= top["bic"] and (better is None or g["bic"] < better[0]["bic"]):
                    better = (g, r)
            if better is None:
                break
            path.append(f"With site blocking allowed, '{better[1].text()}' is no longer needed (BIC {top['bic']:.0f} -> "
                        f"{better[0]['bic']:.0f}): the plain search had added it to imitate the site blocking.")
            final.sort(key=lambda f: f["bic"])
        return final

    # ---- where the search starts
    net0 = [] if start is None else parse_network(start)
    cur = best_of([("start", net0)], None, "start", random_start=True)[0][0] if net0 else keep([], _empty_fit(data))
    seen_txt = ", ".join(f"{s} (up to {v:.1f} mol-%)" for s, v in det.items()) or "no product at all"
    path.append(f"The feed is {' and '.join(fed)}; the GC sees these products: {seen_txt}.")
    if net0:
        path.append(f"Start: the network you gave ({'; '.join(r.text() for r in net0)}): {cur['verdict']} (BIC {cur['bic']:.0f}).")
    else:
        path.append(f"Start: no reactions (the outlet would equal the feed), BIC {cur['bic']:.0f}.")
    net, cur, stopped = grow(net0, cur)
    final = finish(cur)

    # ---- if nothing found fits: the search changes one reaction at a time, so a wrong first
    # reaction can lead it into a dead end.  Search again from somewhere else.
    restarts, from_nothing = 0, not net0
    while det and not final[0]["fits_noise"] and not stopped and restarts < n_restarts and not out_of_time():
        if not from_nothing:
            from_nothing, alt, where = True, [], "no reactions at all"
        else:
            alt = None
            for f in sorted((f for f in cache.values() if f["n_reactions"] == 1), key=lambda f: f["bic"]):
                r = parse_reaction(f["reactions"][0])
                if r.entry_key() not in firsts and set(r.reactants) <= set(fed) and r.default_orders:
                    alt, where = [r], f"'{r.text()}' as the first reaction"
                    break
            if alt is None:
                break
            firsts.add(alt[0].entry_key())
        restarts += 1
        # The first new start is searched like the first pass (cheap; it cures a wrong first reaction).
        # If that fails too, the rate laws are suspected: later starts also try every reaction they
        # may add with one of its orders changed (several times more fits per step).
        wide = WIDE_RESTARTS and restarts >= 2
        path.append(f"The best network so far does not fit within the GC noise. The search changes one reaction at a "
                    f"time, so it may have taken a wrong turn early on: search again, starting from {where}."
                    + (" This time every reaction that may be added is also tried with one of its orders changed: so far the "
                       "search took each rate to follow its equation until the reactions were settled, and that may be what "
                       "leads it astray." if wide else ""))
        say("search", path[-1], final[0])
        net, cur, stopped = grow(alt, cache.get(network_key(alt)) or keep(alt, _empty_fit(data)), wide=wide)
        final = finish(cur)
    final = _add_ranks(final)[:n_final + 2]
    # exactly tied networks (same BIC to 0.5): write the one with fewer two-way reactions first
    n_rev = lambda f: sum(parse_reaction(t).reversible for t in f["reactions"])
    tie = [f for f in final if f["delta_bic"] < 0.5]
    if len(tie) > 1:
        tie.sort(key=lambda f: (n_rev(f), f["bic"]))
        final = tie + final[len(tie):]
        for i, f in enumerate(final, 1):
            f["rank"] = i
            f["delta_bic"] = max(float(f["bic"] - final[0]["bic"]), 0.0)
    for i, f in enumerate(final, 1):
        f["name"] = "best network" if i == 1 else f"runner-up {i - 1}"
    best = final[0]
    for f in final[1:]:
        f["differs_by"] = _difference(best["reactions"], f["reactions"])
        if f["model"] != best["model"] and "blocks catalyst sites" not in f["differs_by"]:
            f["differs_by"] += "; and whether product C blocks catalyst sites"
    path.append(f"Result: {'; '.join(best['reactions'])} with {best['model_words']} (BIC {best['bic']:.0f})." if best["reactions"]
                else "Result: no reaction.")
    tied = [f["name"] for f in final[1:] if f["tied_with_best"]]
    undecided = [f"The data cannot decide {f['differs_by']} (BIC difference {f['delta_bic']:.1f}, below {INDISTINGUISHABLE:g})."
                 for f in final[1:] if f["tied_with_best"]]
    complete = not best["not_made"]
    if best["reactions"]:
        summary = ["The reactions that explain the data best:"] + ["   " + e for e in best["equations"]]
        summary.append(f"This network {best['verdict']} ({best['model_words']}).")
    elif best["fits_noise"]:
        summary = ["No reaction is needed to explain these data: the GC reading equals the feed within the GC noise "
                   "in every run (nothing reacts at these conditions)."]
    else:
        summary = ["No reaction could be proposed: the GC reading differs from the feed, but no product is seen clearly "
                   "enough above the GC noise to say what is made."]
    if not complete:
        summary.append("Warning: " + " and ".join(best["not_made"]) + (" is" if len(best["not_made"]) == 1 else " are")
                       + " seen by the GC but made by no reaction of this network (the search stopped before it "
                       "could explain everything, or the library holds no reaction that does).")
    if not best["fits_noise"]:
        summary.append("Even the best network leaves misses larger than the GC noise: either something is happening that "
                       "the library of reactions and rate laws does not describe (for example an adsorption effect), or "
                       "the step-by-step search did not find the right combination. If you have an idea of the "
                       "reactions, type it as a candidate and rank it against this result.")
    if undecided:
        summary += undecided
    elif len(final) > 1 and best["fits_noise"] and final[1]["delta_bic"] < CLEAR_GAP:
        summary.append(f"The next best network tried is ruled out, but only just (BIC difference {final[1]['delta_bic']:.1f}; "
                       f"below {INDISTINGUISHABLE:g} would count as a tie): a few more runs would make this firm.")
    elif len(final) > 1 and best["fits_noise"]:
        summary.append(f"No other network tried comes close (next best: BIC difference {final[1]['delta_bic']:.0f}).")
    stopped = stopped or out_of_time()
    if stopped:
        summary.append(f"The search was stopped by the time limit ({max_seconds:g} s): this is the best network found so "
                       "far, not the end of the search.")
    notes, cautions = list(data.get("notes", [])), _cautions(data, best["n_params"])
    summary += notes + cautions
    return dict(best=best, runners_up=final[1:], tied=tied, decided=not tied, undecided=undecided, path=path,
                complete=bool(complete), n_fits=int(count[0]), seconds=float(time.time() - t0), library_size=len(lib),
                detected=det, stopped_early=bool(stopped), restarts=int(restarts), n_runs=int(len(data["gc"])),
                n_settings=n_settings(data), notes=notes, cautions=cautions, summary=[s for s in summary if s])


# =================================================================================== one step of the search, on its own
def one_step_revision(network, data, library=None, theta=None, max_reactions=4, order_choices=ORDER_CHOICES,
                      n_polish=3, inhib=None):
    """Every SINGLE change of a network, fitted to the data, best first.

    This is one step of the automatic search (discover), offered on its own so that the
    advisor (advisor.SmartAdvisor with revise=True) can repair its reaction network DURING a
    campaign, one well-supported change at a time, on the campaign's own runs.

    network        the network as it stands (text, lines, Reaction objects or a ReactionSystem)
    data           the run log (load_data / make_data)
    library        reaction_library() by default (enumerated from the atom balance)
    theta          fitted kinetics of the network [ln k_1, Ea_1/(R T_ref), ...] to start from
                   (default: the generic starting guess)
    max_reactions  no reaction is added to a network that already has this many
    n_polish       how many of the best changes are fitted to convergence

    The changes tried (the same four kinds the search uses):
      add      a library reaction whose reactants are fed or already made.  While a product the GC
               sees is made by no reaction, only reactions that make such a product; otherwise only
               reactions that touch a species the network misses by more than the GC noise allows
      remove   one reaction (if the others can still run from the feed)
      order    one reactant order of one reaction set to another value of order_choices (0, 1, 2)
      replace  one reaction by another variant of itself (the other direction, one way / both ways)
               or by another library reaction that makes the same product
    All changed networks are first compared by a quick fit in one batch (plain rate laws,
    N_SCREEN iterations), each from TWO starting points: the present fit (reactions kept keep
    their constants) and the generic starting guess.  The second matters when the network as it
    stands is wrong: its constants have been bent to make up for what it lacks (seen with C + B
    -> E missing: from the bent constants the right repair stopped at 8 times the GC noise, from
    the generic guess it reached 0.6).  The better of the two is kept, and the best n_polish
    changes are then fitted to convergence.
    inhib (optional): K_C at the temperature of every run (n,), 1/bar, from the caller's own fit
    when that fit found product C blocking catalyst sites.  Every network (the present one and
    every change) is then ALSO screened with site blocking at that strength (not fitted again,
    so no extra parameter), and the better of the two footings is kept.  Without it the screen
    uses plain rate laws only, and on a reactor with site blocking a wrong change can then
    imitate the site blocking and push the right change out of the few the caller compares
    (seen on a 10-run campaign: making A + B -> C two-way, the right repair, came 4th of 47
    with plain rate laws, behind 'zero order in A', which mostly stood in for site blocking).
    Nothing is decided here: the caller compares the best changes with the network as it stands
    (like for like, with its own kinetic model classes) and decides whether the evidence is
    strong enough.

    Returns a dict (JSON-friendly):
      current   the fit_network dict of the network as given (converged; plain rate laws, or site
                blocking at the given strength when that fits better)
      moves     the changed networks, best (lowest BIC) first, each a dict:
                  kind ('add' | 'remove' | 'order' | 'replace'), what (the change in plain words),
                  reactions [text], fit (fit_network dict), converged (bool: among the n_polish
                  fitted to convergence), gain (BIC of the present network minus BIC of the changed
                  one; positive = the change explains the data better), site_blocking (True if the
                  fit kept uses the given site blocking, False for plain rate laws)
      n_tried, seconds, detected {species: largest mol-% seen}"""
    t0 = time.time()
    net = parse_network(network)
    lib = _library_reactions(library)
    fed, det = fed_species(data), detected_species(data)
    visible = set(fed) | set(det)
    lib = [r for r in lib if set(r.products) <= visible or (r.reversible and set(r.reactants) | set(r.products) <= visible)]
    # the footings every network is fitted on: plain rate laws, and (if given) site blocking at the caller's strength
    feet = [None] if inhib is None else [None, np.broadcast_to(np.asarray(inhib, float), (len(data["gc"]),)).copy()]

    def fit_all(nets_, start_sets, n_iter, tol):
        """Every network from every start set (a list of start vectors per set) on every footing, in
        ONE batch; for each network the state of lowest cost and the footing it came from."""
        rows = [(i, fi) for fi in range(len(feet)) for sts in start_sets for i in range(len(nets_))]
        states = _batch_lm([nets_[i] for i, _ in rows], data,
                           [sts[i] for fi in range(len(feet)) for sts in start_sets for i in range(len(nets_))],
                           n_iter, tol=tol, inhib=None if inhib is None else [feet[fi] for _, fi in rows])
        best = [None] * len(nets_)
        for (i, fi), st in zip(rows, states):
            if best[i] is None or st["cost"] < best[i][0]["cost"]:
                best[i] = (st, fi)
        return best

    def scored(n_, st_fi):
        f = _state_fit(n_, st_fi[0], data)
        f["site_blocking"] = bool(st_fi[1])
        if st_fi[1]:
            f["model_words"] = "rate laws with product C blocking catalyst sites at the strength the caller's own fit found"
        return f

    if net:
        th0 = generic_theta(len(net)) if theta is None else np.asarray(theta, float)[:2 * len(net)]
        cur = scored(net, fit_all([net], [[th0]], 20, 1e-6)[0])
        th_cur = np.asarray(cur["theta"], float)
    else:
        cur, th_cur = _empty_fit(data), np.zeros(0)
    have = set(fed) | _made_by(net)
    entries = {r.entry_key() for r in net}
    orphan = set(cur["not_made"])
    poor = {s for s, m in cur["miss"].items() if m["x_noise"] > 1.3} | orphan
    moves = {}

    def offer(kind, what, new):
        key = network_key(new)
        if key != network_key(net) and key not in moves and _reachable(new, fed):
            moves[key] = (kind, what, new)

    if len(net) < max_reactions and poor:
        for r in lib:
            makes = set(r.products) | (set(r.reactants) if r.reversible else set())
            touches = (makes & orphan) if orphan else ((set(r.reactants) | set(r.products)) & poor)
            starts_ok = set(r.reactants) <= have or (r.reversible and set(r.products) <= have)
            if r.entry_key() not in entries and starts_ok and touches:
                offer("add", f"add '{r.text()}'", net + [r])
    for j, r in enumerate(net):
        rest = net[:j] + net[j + 1:]
        if rest:
            offer("remove", f"remove '{r.text()}'", rest)
        for q in _order_variants(r, order_choices):
            offer("order", f"change the rate law of '{r.equation()}' to {q.rate_law().strip()} (was {r.rate_law().strip()})",
                  net[:j] + [q] + net[j + 1:])
        others, made = entries - {r.entry_key()}, set(r.products) & set(det)
        for q in lib:
            twin = q.entry_key() == r.entry_key()
            if q.key() != r.key() and q.entry_key() not in others and (twin or set(q.products) & made):
                offer("replace", f"replace '{r.text()}' by '{q.text()}'", net[:j] + [q] + net[j + 1:])
    out = []
    if moves:
        keys = list(moves)
        nets = [moves[k][2] for k in keys]
        # every change from the present fit and from the generic guess (on every footing), one batch
        best = fit_all(nets, [[_warm_theta(n_, net, th_cur) for n_ in nets], [generic_theta(len(n_)) for n_ in nets]],
                       N_SCREEN, 1e-4)
        fits = [scored(n_, b) for n_, b in zip(nets, best)]
        order = sorted(range(len(keys)), key=lambda i: fits[i]["bic"])
        top = order[:n_polish]
        pol = _batch_lm([nets[i] for i in top], data, [np.asarray(fits[i]["theta"], float) for i in top], 20, tol=1e-6,
                        inhib=None if inhib is None else [feet[best[i][1]] for i in top])
        for i, st in zip(top, pol):                    # each polished on the footing that won its screen
            f = scored(nets[i], (st, best[i][1]))
            if f["bic"] < fits[i]["bic"]:
                fits[i] = f
        for i in sorted(range(len(keys)), key=lambda i: fits[i]["bic"]):     # the converged ones come first: they were the best
            kind, what, new = moves[keys[i]]
            out.append(dict(kind=kind, what=what, reactions=[r.text() for r in new], fit=fits[i], converged=i in top,
                            gain=float(cur["bic"] - fits[i]["bic"]), site_blocking=fits[i]["site_blocking"]))
    return dict(current=cur, moves=out, n_tried=len(out), seconds=float(time.time() - t0), detected=det)


# =================================================================================== the run that tells two networks apart
def _model_of(fit):
    """A fitted network (fit_network dict) -> (KineticFit set to the fitted values, theta, cov)."""
    S = build_system(fit["reactions"], name=fit.get("name", "network"))
    kf = KineticFit(S, a_spec=fit["a_spec"], models=(fit["model"],))
    kf.model, kf.theta = fit["model"], np.asarray(fit["theta"], float)
    # a result saved without its covariance (discover_reactions.strip) still predicts; it only has no uncertainty
    kf.cov = np.asarray(fit["cov"], float) if fit.get("cov") else np.zeros((len(kf.theta), len(kf.theta)))
    return kf


def predict_gc(fit, phys):
    """Mole fractions (n, 6) a fitted network (a fit_network dict) predicts at operating rows
    phys (n, 6): P bar, V_A, V_B, V_I mL/min, T kelvin, catalyst g."""
    kf = _model_of(fit)
    F, _ = kf.simulate(np.atleast_2d(np.asarray(phys, float)), kf.theta, model=fit["model"])
    return F / F.sum(1, keepdims=True)


def _pair(networks_fitted):
    """The two fitted networks to tell apart, from a list, a rank_candidates result or a discover result."""
    x = networks_fitted
    if isinstance(x, dict) and "ranking" in x:
        x = x["ranking"]
    elif isinstance(x, dict) and "best" in x and "runners_up" in x:
        x = [x["best"]] + list(x["runners_up"])
    x = [f for f in x if f.get("theta")]
    if len(x) < 2:
        raise ValueError("Two fitted networks are needed to plan a run that tells them apart.")
    return x[0], x[1]


def discriminating_run(networks_fitted, data, x_last=None, max_step=None, lo=None, hi=None, wcat=None,
                       n_candidates=2048, n_draws=6, seed=0):
    """The next experiment that would best tell the top two networks apart.

    networks_fitted  two fit_network dicts, or the result of rank_candidates / discover (the
                     best two are taken)
    data             the run log (for the catalyst mass and for comparison with the runs done)
    x_last           the previous run [P bar, V_A, V_B, V_I mL/min, T kelvin]; when given, the new
                     run stays within one allowed step (max_step, default systems.MAX_STEP) of it
    lo, hi           the box in the same units (default: the operating box systems.LOW / HIGH)
    wcat             catalyst mass in g (default: that of the last run in the log; never a knob)

    Both fitted networks predict the GC composition at many operating points inside the box
    (Sobol points, then two rounds zooming in on the best).  The point returned is where the
    two predictions differ most, counted in units of the GC noise and allowing for how
    uncertain each network's own prediction is (parameter draws from each fit):
        score = sum over species of (y1 - y2)^2 / (noise^2 + var1 + var2)
    Set points are rounded to what can be set (0.5 bar, 0.5 mL/min, 1 deg C).
    Far from the runs done, two fits can predict very different readings only because their
    rate constants are still uncertain there; such a difference would shrink once the run is
    made and both networks are fitted again.  The uncertainty-aware number is therefore the one
    to plan with, and the message and expected_bic_gap are based on it.

    Returns a dict (JSON-friendly): run {pressure_bar, ..., temperature_C, catalyst_g} (the CSV
    columns), x [P, V_A, V_B, V_I, T_K], networks [the two names], predicted {name: {species:
    mol-%}}, difference_x_noise {species: difference / GC noise}, separation (overall difference
    of the two best-fit predictions in units of the GC noise), separation_uncertain (the same
    allowing for the uncertainty of the fitted numbers of both networks; equal to separation
    when the fits carry no covariance), expected_bic_gap (separation_uncertain^2: about what
    one such run adds to the BIC difference), best_so_far (largest best-fit separation among
    the runs already in the log), step_limited (bool), message (plain words)."""
    f1, f2 = _pair(networks_fitted)
    k1, k2 = _model_of(f1), _model_of(f2)
    lo = LOW.copy() if lo is None else np.asarray(lo, float)
    hi = HIGH.copy() if hi is None else np.asarray(hi, float)
    lo, hi = np.maximum(lo, LOW), np.minimum(hi, HIGH)
    limited = x_last is not None
    outside = False
    if limited:
        x_last = np.asarray(x_last, float)[:5]
        step = MAX_STEP if max_step is None else np.asarray(max_step, float)
        # the box comes first: if the last run itself lies outside it, the new run is the nearest
        # point of the box, even where that is more than one step away
        outside = bool(np.any(x_last < lo - 1e-9) or np.any(x_last > hi + 1e-9))
        lo, hi = np.clip(x_last - step, lo, hi), np.clip(x_last + step, lo, hi)
    W = float(data["phys"][-1, 5]) if wcat is None else float(wcat)
    grid, off = np.array([0.5, 0.5, 0.5, 0.5, 1.0]), np.array([0.0, 0.0, 0.0, 0.0, 273.15])   # whole deg C
    rng = np.random.default_rng(seed)
    sp = data["species"]

    def rows(X):
        return np.column_stack([X, np.full(len(X), W)])

    def sep(X, draws=0):
        """(score, y1, y2) at physical points X; with draws > 0 the score allows for parameter uncertainty."""
        out = []
        for kf in (k1, k2):
            F, _ = kf.simulate(rows(X), kf.theta, model=kf.model)
            y = F / F.sum(1, keepdims=True)
            var = np.zeros_like(y)
            if draws:
                p = len(kf.theta)
                try:
                    L = np.linalg.cholesky(kf.cov + 1e-12 * np.eye(p))
                except np.linalg.LinAlgError:
                    w, V = np.linalg.eigh(kf.cov)
                    L = V * np.sqrt(np.maximum(w, 0))
                th = kf.theta + rng.standard_normal((draws, p)) @ L.T
                ys = []
                for t in th:
                    Fd, _ = kf.simulate(rows(X), t, model=kf.model)
                    ys.append(Fd / Fd.sum(1, keepdims=True))
                var = np.nanvar(np.array(ys), axis=0)
            out.append((y, var))
        (y1, v1), (y2, v2) = out
        sig = gc_noise(0.5 * (y1 + y2))
        s = np.sum(((y1 - y2) ** 2 / (sig ** 2 + v1 + v2))[:, :-1], axis=1)
        return np.nan_to_num(s, nan=0.0), y1, y2

    def feasible_points(X):
        X = np.clip(np.round((X - off) / grid), np.ceil((lo - off) / grid - 1e-9), np.floor((hi - off) / grid + 1e-9)) * grid + off
        X = np.clip(X, lo, hi)
        return X[X[:, 1] + X[:, 2] + X[:, 3] >= MIN_TOTAL_FLOW - 1e-9]

    u = qmc.Sobol(5, scramble=True, seed=seed).random(n_candidates)
    X = lo + u * (hi - lo)
    X[: n_candidates // 2, 0] = np.exp(np.log(lo[0]) + u[: n_candidates // 2, 0] * (np.log(hi[0]) - np.log(lo[0])))
    X = feasible_points(np.vstack([X, lo, hi, 0.5 * (lo + hi)] + ([x_last] if limited else [])))
    if len(X) == 0:
        raise ValueError("No operating point in the box leaves enough total flow for the GC.")
    s, _, _ = sep(X)
    xb = X[np.argmax(s)]
    for width in (0.12, 0.04):                               # zoom in on the best point
        u = qmc.Sobol(5, scramble=True, seed=seed + 1).random(256)
        Z = feasible_points(np.vstack([xb + (2 * u - 1) * width * (hi - lo), xb]))
        sz, _, _ = sep(Z)
        X, s = np.vstack([X, Z]), np.r_[s, sz]
        xb = X[np.argmax(s)]
    # the points where the best-fit predictions differ most are scored again allowing for the
    # uncertainty of both fits; the run proposed is the best by THAT score
    top = X[np.argsort(-s)[:N_UNCERTAIN]]
    top = np.unique(top, axis=0)
    has_cov = bool(np.any(k1.cov) or np.any(k2.cov))
    su, _, _ = sep(top, draws=n_draws if has_cov else 0)
    x = top[np.argmax(su)]
    # the score that picked x is the largest of many rough ones (few draws), so it flatters x:
    # the number reported is worked out again at x alone, from fresh and more numerous draws
    su_x, _, _ = sep(x[None, :], draws=4 * n_draws if has_cov else 0)
    s1, y1, y2 = sep(x[None, :])
    sig = gc_noise(0.5 * (y1 + y2))
    dz = ((y1 - y2) / sig)[0]
    done, _, _ = sep(data["phys"][:, :5])
    names = [f1.get("name", "network 1"), f2.get("name", "network 2")]
    if names[0] == names[1]:
        names = ["network 1", "network 2"]
    big = sorted(range(len(sp) - 1), key=lambda i: -abs(dz[i]))[:2]
    run = dict(zip(LOG_COLS, [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(round(x[4] - 273.15, 2)), W]))
    separation = float(np.sqrt(s1[0]))
    sure = min(float(np.sqrt(su_x[0])), separation)          # never more than the best-fit difference itself
    words = (f"Run at {x[0]:g} bar, {x[1]:g} / {x[2]:g} / {x[3]:g} mL/min of A / B / inert and {x[4] - 273.15:.0f} C"
             + (" (the last run lies outside the operating box; this run is inside the box, as close to one allowed step "
                "from it as the box permits)" if outside else " (within one allowed step of the last run)" if limited else
                " (a free run: anywhere in the operating box, not held to one allowed step from the last run)") + ". "
             f"There the fitted '{names[0]}' and '{names[1]}' predict these GC readings: "
             + "; ".join(f"{sp[i]} {100 * y1[0, i]:.2f} against {100 * y2[0, i]:.2f} mol-% ({abs(dz[i]):.0f}x the GC noise)" for i in big)
             + f". Overall the two predictions are {separation:.0f}x the GC noise apart"
             + (f" (the runs done so far reach at most {np.sqrt(done.max()):.0f}x)." if len(done) else "."))
    if has_cov:
        words += (" Each network's own prediction is still uncertain there, because its fitted rate constants are; counted "
                  f"against the GC noise and that uncertainty together, the two predictions are about {sure:.0f}x apart.")
    if sure < 3 and separation >= 3:
        words += (" That is not enough to decide with one run: the two fits differ there mainly because their rate constants "
                  "are not yet pinned down at such conditions, and the difference would shrink once the run is made and both "
                  "networks are fitted again. Several runs" + (" in this direction" if limited else "") + " would be needed.")
    elif sure < 3:
        words += (" That is not enough to decide" + (" within one step; several steps in this direction or a "
                  "free kinetic run would be needed." if limited else ": inside this box the two networks predict almost the same."))
    return dict(run=run, x=[float(v) for v in x], networks=names,
                predicted={n: {s_: float(100 * y[0, i]) for i, s_ in enumerate(sp)} for n, y in zip(names, (y1, y2))},
                difference_x_noise={s_: float(dz[i]) for i, s_ in enumerate(sp[:-1])},
                separation=separation, separation_uncertain=sure,
                expected_bic_gap=sure ** 2, best_so_far=float(np.sqrt(done.max())) if len(done) else 0.0,
                step_limited=bool(limited), message=words)
