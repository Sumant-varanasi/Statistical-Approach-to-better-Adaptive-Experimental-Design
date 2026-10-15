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

The benchmark's reactor uses the true kinetics plus Langmuir–Hinshelwood **site blocking** by
adsorbed product and, more weakly, adsorbed reactant:

$$r_j \to \frac{r_j}{\left(1 + K_C(T)\,P_C + K_A\,P_A\right)^{m_j}}, \qquad
  K_C(T) = K_C(T_{ref})\,\exp\!\left[e\left(\frac{T_{ref}}{T} - 1\right)\right],\ e = \frac{-\Delta H_{ads}}{R\,T_{ref}},$$

with $m_j$ the number of surface sites in the rate-determining step (the molecularity of the
forward reaction: 2 for A + B → C and C + B → E, 1 for A → D). Adsorption is exothermic, so
$K_C$ falls with temperature ($\Delta H_{ads} \approx -48$ kJ/mol for the complex network).
GC noise is 1 % relative + 0.05 % absolute on every mole fraction, renormalised.

Optionally the catalyst **loses activity on stream** (the `complex_deact` scenario): the
active area shrinks by a factor $\exp(-\lambda\,\Delta t)$ during each run of duration
$\Delta t$ hours, with a severity-weighted rate the advisor's model does not know,

$$\lambda = \lambda_0\,\exp\!\left[-\frac{40\ \mathrm{kJ/mol}}{R}\left(\frac1T - \frac1{T_{ref}}\right)\right]\sqrt{\frac{P_A}{4\ \mathrm{bar}}},$$

i.e. coking that is faster when hotter and with more A in the feed ($\lambda_0$ = 2 % per day
at 10 bar, 40 % A, $T_{ref}$).

The advisor's model (section 8.1) can learn product adsorption and its heat and a constant
deactivation rate, but never has the reactant term or the severity dependence: its physics is
imperfect in the way any textbook model is on a real reactor, and the GC-derived profit is noisy.

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
4. 25 runs in all (five weeks at one run per day, 24 h on stream each). Ten campaigns per
   method, each from a different plausible base case (seed 0 is the project's 10 bar,
   20/20/10 mL/min, 600 K).
5. On a deactivating catalyst (`complex_deact`), every method repeats the base case as a
   **check run** every 5th run: the way an operator sees drift, and the only way the model can.

Score: the true profit of the settings the method would *recommend* after each run (its best
run by predicted or measured profit), as a share of the best possible profit at that catalyst
loading (found by differential evolution on the true model).

## 8. The advisor (v5)

### 8.1 Learning the chemistry: a ladder of model classes

The fit (`smart/kinetics.py`) is Bayesian least squares on every GC mole fraction with a wide
prior around a chemist's order-of-magnitude guess; uncertainty from the Laplace approximation,
inflated by $\chi^2/\text{dof}$ when the physics fits worse than the GC noise explains. Five
nested model classes are fitted, each warm-started from the one below it:

| Model | Adds | Parameters |
|---|---|---|
| `plain` | $\ln k_{j,ref}$, $E_{a,j}$ for every reaction | $2n_r$ |
| `inhib` | product adsorption, every rate $/(1 + K_C(T) P_C)^{m_j}$, with the typical adsorption heat ($e = 10$) | +1 ($\ln K_C$) |
| `inhib_T` | the adsorption heat itself | +1 ($e \ge 0$) |
| `decay` | catalyst activity $\exp(-d\,t)$, $t$ = hours on stream before the run ($d<0$ = break-in) | +1 ($d$) |
| `plain_decay` | decay without adsorption | +1 |

The data choose by BIC, $\chi^2_{\text{noise}} + p \ln n$, with $\chi^2$ weighted by the GC's
*own* noise (the fit weights carry an allowance for model error, which would understate the
evidence); the decay parameter counts double, so drift is only claimed when clearly supported.
**Deactivation can only be told apart from the kinetics by repeating a set point later**, so
the decay models are only tried when the log holds such a replicate pair (a check run). The
advisor reports what the data support ("your data show product inhibition, $K_C \approx 0.03$
/bar"; "no set point has been repeated, so drift cannot be told yet").

Measured on the virtual complex reactor (15 runs): the plain model gives $E_a$ = 69 / 119 / 165
kJ/mol (true 70 / 120 / 150) with $\chi^2/\text{dof}$ = 13; the ladder picks `inhib` and gives
71 / 120 / 151 with $\chi^2/\text{dof}$ = 0.10.

### 8.2 Choosing the next run: a receding-horizon planner

1. **Plausible chemistries**: the best estimate plus five posterior draws.
2. **Plan**: predict profit over the whole box (1024 Sobol points + the runs done + local
   refinement) and find the model's optimum $x^*$; report how many runs away it is under the
   step limits (the bottleneck knob decides). With a spec, only points likely to meet it count.
3. **Correct and calibrate**: a Gaussian process (Matérn, length-scales ≥ 0.15, noise floor from
   the GC error propagated to profit) on the residual *measured − physics*; the predictive
   spread is scaled by $c = \max(1, \text{RMS}\,z)$ of the advisor's own past misses at the data.
4. **First steps within reach**: the step toward $x^*$, the corners of the step box (full steps
   on every free knob), points on the line to $x^*$, "stay", and Sobol points; candidates
   unlikely to meet the spec ($P_{\text{feas}} < 0.5$) or below the flow minimum are dropped.
   Shortlist: top-6 by Expected Improvement, top-3 by predicted mean, the step toward $x^*$, stay.
5. **Roll forward**: from each shortlisted step, follow the model mean greedily for $H-1$
   further allowed steps (batched over all shortlisted steps per depth).
6. **Score the whole trajectory under every plausible chemistry** (plus a GP residual draw):
   $$\text{return} = \sum_{h=1}^{H} f_\theta(x_h) + (R - H)\,\max\!\big(\text{best},\ \max_h f_\theta(x_h)\big),$$
   $R$ = runs the operator can still afford, $H = \min(4, R)$; choose the first step of the
   trajectory with the highest expected return per remaining run. A step that earns nothing
   now but opens a better region wins when there are enough runs left to cash it in; a safe
   step wins when there are not. "Stay" wins when nothing within reach is worth the risk.
7. **Recovery**: when nothing within one step is likely to meet the spec, take the step most
   likely to get back within spec that also heads toward $x^*$.
8. Recommend the tested run with the highest *predicted* profit among those that met the
   spec as measured (a lucky GC reading cannot fool it).

The v4 rule (Expected Improvement one step ahead with the purposeful step toward $x^*$) is
kept as `planner="greedy"` for comparison.

### 8.3 Product and safety specs

`spec = {"S_C": 0.9, "dT_ad": 60}` means selectivity to C at least 0.9 and adiabatic
temperature rise at most 60 K; `"X_A"` (minimum conversion) is also available. The
adiabatic rise $\Delta T_{ad} = \sum_j (-\Delta H_j)\,\xi_j / (F_{T,0}\,c_p)$ is the temperature
the reaction would reach if the heat were not removed: a heat-load indicator the isothermal
model cannot see. For this chemistry it is hundreds of kelvin at any useful conversion, so it
is informative rather than a sensible limit. Feasibility of a candidate is the fraction of the
plausible chemistries under which it meets the spec.

Cost: 5–15 s per suggestion (a few thousand reactor simulations with the vectorised
Rosenbrock solver in `smart/systems.py`; the 5-model fit is warm-started after the first).

Score: the true profit of the settings the method would *recommend* after each run, at the
catalyst's current activity, as a share of the best possible profit on a fresh catalyst; and
the **cumulative profit earned** over the campaign (every run produces, so a campaign that
wanders earns less).

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
- **Smart v4**: the previous advisor (plain kinetics, one-step Expected Improvement with the
  purposeful step), to measure what the v5 kinetics ladder and planner add.
- **Stress test**: the v5 advisor with the over-reaction C + B → E removed from its model (E still
  appears in the GC; only the GP correction can account for the lost product).

## 10. What is still placeholder

Kinetics, prices, GC noise, the inhibition and the step limits are plausible numbers, not
measured ones. The next step is a log from the Micromeritics reactor: the advisor reads it
as is (`reactor_advisor.py my_log.csv`).
