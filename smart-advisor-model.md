# Smart reactor advisor: model and protocol

This note extends the reactor model in `digital-twin-fixed-bed-reactor.md` (Prof. Khare's
original formulation, unchanged) with what the smart advisor adds. Sections 1–4 are physics,
5–7 are how a campaign is run, 8–9 are the advisor and the baselines it is compared with.

## 1. Catalyst surface area as the reactor coordinate

The reaction takes place on the catalyst surface. Instead of integrating the packed-bed mole
balances over catalyst *mass* W, the advisor integrates over catalyst *surface area*

$$S = a_s\,W_{\mathrm{cat}}, \qquad a_s = \text{specific surface area (BET), m}^2\,\text{g}^{-1}$$

$$\frac{dF_i}{dS} = \sum_j \nu_{ij}\, r_j, \qquad r_j\ \big[\mathrm{mol\,m^{-2}\,s^{-1}}\big], \qquad 0 \le S \le a_s W_{\mathrm{cat}}.$$

Rate constants are therefore per m² of catalyst surface:

$$r_1 = k_1(T)\,P_A P_B \quad [\mathrm{mol\,m^{-2}\,s^{-1}\,bar^{-2}}], \qquad
  r_2 = k_2(T)\,P_A \quad [\mathrm{mol\,m^{-2}\,s^{-1}\,bar^{-1}}].$$

Consequences:

- The kinetic parameters are properties of the catalyst **material**, not of the amount loaded.
  A fit made with one loading transfers to another loading or to a batch with a different BET
  area. (Checked: 0.5 g at 200 m²/g gives the same outlet as 1 g at 100 m²/g.)
- With $a_s = 100\ \mathrm{m^2/g}$ and 1 g of catalyst the model is identical to the per-gram
  model in `reactor_twin.py` (checked to 1e-9).
- Numerically the bed is integrated in the dimensionless coordinate $s = S/(a_s W_{\mathrm{cat}})$
  with flows divided by the inlet total flow, so the solver sees O(1) states regardless of
  loading or feed.

The operator enters $a_s$ once per catalyst batch (`--surface-area`). In the virtual reactor
the same relation holds, so the benchmark tests the whole chain.

## 2. The inert as a knob

The inert flow $\dot V_{I,0}$ is a set point like the others. It enters only through the total
molar flow $F_T$ and so through every partial pressure $P_i = P\,F_i/F_T$: more inert dilutes
the feed and slows every reaction. It also serves as the GC internal standard:
$F_{I,\mathrm{out}} = F_{I,0}$, so every outlet molar flow follows from the GC reading alone,

$$F_{i,\mathrm{out}} = y_{i,\mathrm{out}}\,\frac{F_{I,0}}{y_{I,\mathrm{out}}}.$$

For that the inert must be measurable: $\dot V_{I,0} \ge 2$ mL/min.

## 3. The two reaction networks

| | simple (project network) | complex |
|---|---|---|
| reactions | A + B → C, A → D | A + B ⇌ C (reversible, ΔH = −120 kJ/mol, ΔS = −170 J/mol/K), C + B → E, A → D |
| rates | $k_1 P_A P_B$, $k_2 P_A$ | $k_1\,(P_A P_B - P_C/K_{eq}(T))$, $k_2 P_C P_B$, $k_3 P_A$ |
| equilibrium | – | $\ln K_{eq} = -\Delta H/RT + \Delta S/R$ (thermodynamics, not fitted) |
| unknowns the advisor learns | $k_{1,ref}, E_{a1}, k_{2,ref}, E_{a2}$ | six: $k_{j,ref}, E_{aj}$ for $j = 1..3$ |

Parameters are stored as $a_j = \ln k_j(T_{ref})$ and $b_j = E_{aj}/(R\,T_{ref})$ with
$T_{ref} = 573.15$ K, so $\ln k_j(T) = a_j - b_j\,(T_{ref}/T - 1)$.

## 4. The virtual "real" reactor

The benchmark's reactor uses the true kinetics **plus two things the advisor's model does not
contain**: product inhibition, $r_j \to r_j/(1 + K_C P_C)^2$, and GC noise (1 % relative +
0.05 % absolute on every mole fraction, renormalised). So the advisor's physics is imperfect in
the way any textbook model is on a real reactor, and the GC-derived profit is noisy.

## 5. Knobs, campaign settings and constraints

| | symbol | range | changed between runs? |
|---|---|---|---|
| pressure (BPR) | P | 1–100 bar | yes, ≤ 10 bar per run |
| flow of A, B (MFC) | $\dot V_A, \dot V_B$ | 1–50 mL/min | yes, ≤ 10 mL/min per run |
| inert flow (MFC) | $\dot V_I$ | 2–50 mL/min | yes, ≤ 10 mL/min per run |
| temperature | T | 450–850 K | yes, ≤ 30 K per run |
| catalyst mass | $W_{cat}$ | loaded once (default 1 g) | **no** (unloading the bed) |
| BET area | $a_s$ | per batch (default 100 m²/g) | no |
| total feed | | ≥ 10 mL/min | for the GC / BPR |

The step limits (`MAX_STEP`, `--max-step`) model what the hardware and good practice allow
between consecutive runs: furnace ramping, pressure equilibration, keeping the catalyst in a
known state. Every method in the benchmark, including the baselines, obeys them.

## 6. The economics

Profit per hour (placeholder prices, flows in mmol/h):

$$\Pi = v_C F_C - c_{feed}(F_{A,0} + F_{B,0}) - c_{inert} F_{I,0} - \sum_{\text{by-products}} c_s F_s
      - c_{comp}\,F_{T,0}\ln P - c_{cat} W_{cat}.$$

With the catalyst fixed the last term is a constant. Everything is computed from the GC
reading via the internal standard, so the operator supplies nothing but set points and GC
printout.

## 7. How a campaign is run (the protocol every method follows)

1. Load the catalyst ($W_{cat}$, $a_s$).
2. **Five screening runs** by the operator, from the literature/base conditions $(P_0, \dot V_{A0},
   \dot V_{B0}, \dot V_{I0}, T_0)$: a rising temperature ladder $T_0 - 25, T_0, T_0 + 25$ K, then
   $P_0 + 10$ bar, then $\dot V_{B0} + 10$ mL/min. Consecutive runs differ by at most one step.
   These five runs are the same for every method.
3. From run 6 on, the method proposes the next run, which must lie within one step of the
   previous run for every knob, inside the safety box, and satisfy the flow minimum.
4. 25 runs in all (five weeks at one run per day). Ten campaigns per method, each from a
   different plausible base case (seed 0 is the project's 10 bar, 20/20/10 mL/min, 600 K).

Score: the true profit of the settings the method would *recommend* after each run (its best
run by predicted or measured profit), as a share of the best possible profit at that catalyst
loading (found by differential evolution on the true model).

## 8. The advisor

Each round:

1. **Fit the kinetics** to all GC readings so far by Bayesian least squares with a wide prior
   around a chemist's order-of-magnitude guess (`smart/kinetics.py`); uncertainty from the
   Laplace approximation, inflated by $\chi^2/\text{dof}$ when the physics fits worse than the
   GC noise explains.
2. **Draw plausible chemistries**: the best estimate plus five draws from the posterior.
3. **Plan**: predict profit over the whole box (1024 Sobol points + local refinement) under every
   chemistry; the model's optimum $x^*$ is where the best-estimate mean is highest. Report how
   many runs away it is under the step limits (the bottleneck knob decides).
4. **Correct**: a Gaussian process (Matérn, length-scales ≥ 0.15 in scaled units, noise floor
   from the GC error propagated to profit) fitted to the residual *measured − physics* at the
   runs done so far, added to every prediction.
5. **Choose within reach**: candidates in the step box (512 Sobol + the step toward $x^*$ +
   local points), scored by Expected Improvement over the best predicted profit so far.
   If nothing within reach looks like an improvement but the model is confident
   ($\mu^* - 1.64\sigma^* >$ best) that a better region exists, take the **purposeful step**:
   a full step on the bottleneck knob, with the other knobs set to the best predicted values
   at that step.
6. Recommend the tested run with the highest *predicted* profit (a lucky GC reading cannot
   fool it).

Cost: 5–10 s per suggestion (a few thousand reactor simulations with the vectorised
Rosenbrock solver in `smart/systems.py`).

## 9. Baselines

- **Chemist's rules of thumb** (`ChemistPolicy`): read conversion and selectivity off the GC;
  selectivity poor → cooler with more pressure, then more B, then dilute; conversion poor →
  hotter with more pressure, then less flow, then less inert; both fine → more throughput,
  then pressure, then hotter. Momentum (repeat what worked), restart the rules from a new
  best, never repeat an experiment, random allowed step when every rule has been tried. Full
  steps within the limits.
- **Plain BO**: the v2 fitted GP (Matérn, ARD, log P, rank output transform) + Expected
  Improvement on the profit numbers only, candidates within the step box.
- **Random walk**: a random allowed step each run.
- **Stress test**: the advisor with the over-reaction C + B → E removed from its model (E still
  appears in the GC; only the GP correction can account for the lost product).

## 10. What is still placeholder

Kinetics, prices, GC noise, the inhibition and the step limits are plausible numbers, not
measured ones. The next step is a log from the Micromeritics reactor: the advisor reads it
as is (`reactor_advisor.py my_log.csv`).
