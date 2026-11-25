# Smart reactor advisor: model and protocol

This note extends the reactor model in `digital-twin-fixed-bed-reactor.md` (Prof. Khare's
original formulation, unchanged) with what the smart advisor adds. Sections 1–4 are physics,
5–7 are how a campaign is run, 8–9 are the advisor (v5 to v7, with the reaction finder in 8.6 and
the profit vs selectivity trade-off in 8.7) and the baselines it is compared with.

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

## 8. The advisor (v5, v6 in 8.4, v7 in 8.5)

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

### 8.4 v6: a committee for the correction (`smart/committee.py`)

v6 keeps everything above and changes only step 3 of 8.2, the correction of the physics.
The residual $r_i = y_i - \hat f_{\text{phys}}(x_i)$ (measured profit minus the physics prediction
at the best-fit chemistry) is predicted by three members:

| Member | Prediction of $r$ | Joins from |
|---|---|---|
| `none` | $0$, with the GC noise variance $\sigma_n^2$: "the physics is right" | run 1 |
| `gp` | the v5 Gaussian process on the knobs (Matérn 5/2, log P, length-scales ≥ 0.15) | run 4 |
| `nn` | an ensemble of 8 one-hidden-layer tanh networks (16 units, L2 penalty), each on a bootstrap resample, inputs = the knobs (log P) **and** the physics prediction $\hat f_{\text{phys}}(x)$ | run 8 |

The networks see the physics prediction, so they can learn an error that *follows* the physics
(a missing reaction removes a share of the product wherever much product is made); the GP sees
only the knobs and learns errors that vary smoothly across the box. Network outputs are clipped
to the residual range seen plus half of it, so they cannot invent large corrections far from the data.

**Weights by stacking** (Wolpert 1992; Yao et al. 2018). Every member predicts runs it was not
trained on (leave-one-out up to 12 runs, 6-fold after), giving held-out densities
$p_k(r_i)$. The weights maximise the held-out log score of the mixture,
$$\max_{w \in \Delta} \sum_i \log \sum_k w_k\, p_k(r_i) + \sum_k (a_k - 1)\log w_k ,$$
solved by EM, with Dirichlet pseudo-counts $a = (1, 0.5, 0.25)$ for (none, gp, nn): on equal
held-out evidence the simpler member keeps the weight (Occam). The weights are re-estimated
after every run, reported to the operator, and logged per run in the benchmark.

**Prediction**: the mixture mean $\mu = \sum_k w_k \mu_k$ and variance
$\sigma^2 = \sum_k w_k \sigma_k^2 + \sum_k w_k(\mu_k - \mu)^2$: the second term is the members'
**disagreement**. It enters the planner's uncertainty like any other, so the planner is drawn to
where the members disagree (query by committee, Seung et al. 1992).

**Cost**: all networks of all held-out folds are trained in one vectorised L-BFGS run
(`BatchMLP`); a committee fit takes about 1–2 s per run.

In the library the v5 behaviour stays the default (`correction="gp"`), so every v5 result is
reproduced unchanged; v6 is `correction="committee"` (`run_smart(..., correction="committee")`).
The web page and `reactor_advisor.py` used v6 by default until v7 (8.5), which is the default now;
`--advisor v6` / `--advisor v5` or the *Advisor version* setting switch back. Benchmark methods: `smart_v6`, `smart_v6_missing_E`.

**Result** (`python compare_v6.py`, 70 paired campaigns, `results/v6_summary.md`): with the
correct reaction network v6 and v5 are indistinguishable (the committee keeps "none" at about
55–60 % and the correction small). With C + B → E missing from the model v6 is at 65 % of the
best profit after 10 runs against 34 % for v5, reaches 90 % in 14.5 runs against 18.5, and earns
more over the campaign in 8 of 10 campaigns (Wilcoxon p = 0.014). In that case the GP carries
85–90 % of the weight by the end and the networks 5–10 %.

### 8.5 v7: the hybrid committee and network revision

v7 keeps everything above and adds two things, both off by default in the library, so every v5
and v6 result is reproduced unchanged: `correction="hybrid"` and `revise=True`
(`run_smart(..., correction="hybrid", revise=True)`; *Advisor version* v7 on the web page,
`--advisor v7` in `reactor_advisor.py`).

**The hybrid committee** (`smart/committee.py`). A fourth member joins the three of 8.4:

| Member | Prediction of $r$ | Joins from |
|---|---|---|
| `direct` | $\hat y_{\text{NN}}(x) - \hat f_{\text{phys}}(x)$, where $\hat y_{\text{NN}}$ is an ensemble of 8 one-hidden-layer tanh networks (16 units, L2 penalty, each on a bootstrap resample; `ProfitEnsemble`) trained on the **measured profit** $y$ with the knobs (log P) as the only inputs: no physics, no GC composition | run 8 |

Written as a residual it sits on the same footing as the other members. The advisor predicts
$$\hat y(x) = \hat f_{\text{phys}}(x) + \sum_k w_k\,\mu_k(x),$$
so $w_{\text{direct}} = 1$ gives $\hat y = \hat y_{\text{NN}}$ (the physics cancels) and
$w_{\text{direct}} = 0$ gives v6. The member's spread is the spread between its networks. The
weights are the stacking weights of 8.4 with pseudo-counts $a = (1, 0.5, 0.25, 0.25)$ for (none,
gp, nn, direct), and the mixture variance keeps the disagreement term. The networks' output is
clipped to the profits seen plus half their range on each side. Where the physics model gives no
answer (a penalty value) the direct member contributes 0, so it cannot make a failed point look good.

*Held-out evidence.* For `direct` a held-out run is truly unseen: the networks that predict it
are trained without its fold, their targets are scaled with the other runs, and their output is
clipped to the other runs' range. In the v7 committee the GP is therefore also refitted on every
fold, hyper-parameters included. (The v6 committee reuses the full-data hyper-parameters and is
left so, to stay exactly reproducible. That shortcut flatters the GP where the physics is wrong:
on cached campaigns with C + B → E missing its held-out error came out about half of the per-fold
value, 1.98 against 4.50 per hour at 12 runs and 2.08 against 3.84 at 20, while with the right
network the two agree, 1.39 against 1.39 at 20 runs.) The comparison still leans a little toward
the physics-based members, because the kinetic fit behind $\hat f_{\text{phys}}$ has used every
GC reading, the held-out one included (refitting it without the held-out run changed the physics'
own held-out error little: 1.10 → 1.10 per hour with the right network, 26.5 → 26.6 with
C + B → E missing, 12 runs).

**Network revision** (`revise=True`, `SmartAdvisor._revise`). From run 5 on, after the kinetic fit
of every update, the advisor checks its network with the yardstick of 8.6. With the fitted mole
fractions $\hat y_{is}$ and the GC readings $y_{is}$ (runs $i = 1..n$, species $s$ other than the
inert; $N$ readings in all, 5 per run, 4 for the simple system):
$$z_{is} = \frac{\hat y_{is} - y_{is}}{0.01\,y_{is} + 0.0005}, \qquad
  z^{a}_{is} = \frac{\hat y_{is} - y_{is}}{0.02\,y_{is} + 0.002},$$
the first in units of the GC noise, the second in units of the GC noise plus the allowance for
model error with which the kinetic fit itself weighs every reading. With the typical misses
$m_s = \big(\tfrac1n\sum_i z_{is}^2\big)^{1/2}$ and $m^a_s$ likewise, the network **passes** when

1. every product the GC sees (above 3 noise standard deviations in at least two runs) is made by
   some reaction of the network,
2. $m^a_s \le 1$ for every species and $\sum_{i,s} (z^a_{is})^2/(N - p) \le 1$ ($p$ fitted
   parameters): the misses are within the allowance, and
3. $m_s \le 2$ for every species: no species is missed by more than twice the GC noise.

A right network on an imperfect model (the virtual reactor's reactant adsorption, a little ageing)
normally passes; a missing reaction does not. A superfluous reaction can pass.

*One bad reading.* When the check fails, every seen product is made, and one run $k$ carries more
than half of $\sum_{i,s}(z^a_{is})^2$ and more than $3/n$ of it ($n \ge 3$), the network is fitted
again without run $k$. If the other runs then satisfy 1–2 and run $k$ is still missed by more than
the allowance, the advisor names run $k$ and the reading it misses most, asks for it to be
checked, and does not search: a typing error or a bad GC injection shows in one run, a wrong
reaction wherever it matters.

*The search* (`discovery.one_step_revision`). Every single change of the network (8.6: add,
remove, change one order, replace) is screened by a quick batched fit from two starting points,
the present constants and the generic starting guess (constants fitted to a wrong network are bent
to make up for what it lacks). When the advisor's own fit found product C blocking catalyst sites,
every network is also screened with $K_C(T)$ fixed at that strength (no extra parameter), so that
a change that only imitates the site blocking cannot push the right one out. The best 3 changes
are fitted with the advisor's own kinetic ladder (8.1), each from both starting points, and
compared with the present network (fitted once more) by
$$\text{BIC}_w = \chi^2_w + (p + [\text{decay}])\ln N,$$
with $\chi^2_w$ counted with the kinetic fit's weights (GC noise plus allowance), so that the
small misfit a right network leaves on an imperfect model does not count as evidence for another
reaction; the decay parameter counts double, as in 8.1. The best change is adopted when
$$\text{BIC}_w(\text{present}) - \text{BIC}_w(\text{changed}) \ge 10,$$
and $\ge 20$ when the changed network is one the advisor has already left (hysteresis; Kass and
Raftery: a difference above 10 is very strong evidence). Up to 3 changes are made in one update,
each under the same rule. The kinetics of the new network replace the old ones, and the plausible
chemistries (8.2) and the correction are refitted with them. A revised network keeps the heats of
reaction (and, for a two-way reaction, the equilibrium) of every reaction it shares with the
network the advisor was given; new reactions take theirs from the species table (8.6).

*Pacing and safeguards.* The same log is never searched twice. After a search that changed
nothing the advisor waits 3 runs; it makes at most 12 searches and never builds a network of more
than 4 reactions. When no single change clears the bar but the network is still outside the
allowance (a dead end: two wrong reactions can prop each other up), it searches afresh from no
reactions with `discovery.discover` (8.6) on the campaign's own runs, from 8 runs on, at most
twice, each limited to about 60 s, and adopts the result under the same rule. When a run that a
change rested on is edited or removed, every change is taken back and the check starts again from
the network the advisor was given. `revise_seconds` (60 s on the web page) limits the time spent
in one update; it is checked between steps. Every change is logged in plain words
(`SmartAdvisor.revisions`: run, what was wrong, what changed, $\text{BIC}_w$ before and after).
The profit and KPIs of a run come from its GC reading and do not depend on the network.

*Limits.* (i) No step removes a reaction that does no harm: the search only starts when the network
fails the check, so a superfluous reaction, given or added, is kept. (ii) Starting from a very
different network, one change at a time is path dependent and can end at a wrong network; the
fresh search is the way out but is made at most twice and screens with plain rate laws only.
(iii) The time limit is checked between steps, so one batch of fits (5–30 s) can overrun it.
(iv) New reactions take their thermodynamics from the placeholder species table (8.6). (v) Each
search costs 10–60 s.

Benchmark methods (`smart/benchmark.py`, `V7_STUDY`, 18 methods): in every setting `smart_v7`,
`smart_v7_fixednet` (revision off), `nn_only` (section 9), `smart_v6_nonn` and `smart_v6_nogp`
(v6 without its networks / without its GP); with C + B → E missing from the given network,
`smart_v7_missing_E`, `smart_v7_norevise_missing_E`, `smart_v6_nonn_missing_E`,
`smart_v6_nogp_missing_E`; given one of the candidate networks of 8.6 (`close`, `similar`,
`far`), `smart_cand_*` (v5, fixed), `smart_v7_fixednet_cand_*` (v7, fixed) and `smart_v7_cand_*`
(v7, may revise). Results (README section 3.8, `analysis_v7/v7_summary.md`; 10 seeds x 25 runs): with the
right network v7 = v6; with C + B → E missing v7 adds it back by run 6 (median) in 10 of 10 campaigns and earns
200 against v6's 102 (better in 10 of 10, p = 0.002); from a wrong starting network the repair earns more than
v7 with its network fixed in 30 of 30 campaigns. The `direct` member does not help: without the repair, v6 + it
earns less than v6 with C + B → E missing (-35, 10 of 10), and on its own (`nn_only`) a neural network never
reaches 90 % of the best profit on the complex reaction. The web page and `reactor_advisor.py` use v7 by default.

### 8.6 Reaction discovery (`smart/discovery.py`)

**Networks and rate laws.** A network is a list of reactions typed as text (`A + B <=> C`,
`C + B -> E`, `A -> D ; orders A=2`). Reaction $j$ with forward orders $o_{js}$ (default: the
reactant coefficients, mass action) has the rate
$$r_j = k_j(T)\prod_s P_s^{\,o_{js}} \quad\text{(one way)}, \qquad
  r_j = k_j(T)\Big(\prod_s P_s^{\,o_{js}} - \frac{1}{K_j(T)}\prod_s P_s^{\,o'_{js}}\Big) \quad\text{(both ways)},$$
with reverse orders $o'_{js} = o_{js} - \nu^{\text{in}}_{js} + \nu^{\text{out}}_{js}$ (forward orders
plus the stoichiometric coefficients), so that the rate is zero exactly at equilibrium, and
$\ln K_j(T) = -\Delta H_j/RT + \Delta S_j/R$ from the species table below. $k_j(T)$ is Arrhenius
as in section 3, the bed is integrated as in section 1. A reactant of order zero stops driving
the rate as it runs out (factor $y_s/(y_s + 10^{-3})$). Two model classes of the kinetic fit are
used: `plain` (2 parameters per reaction) and `inhib` (+1: every rate divided by
$(1 + K_C(T)P_C)^{m_j}$, $m_j = \sum_s o_{js}$, with the typical adsorption heat of 8.1). The
search fits with `plain`; the best 4 networks are re-fitted with both, and if site blocking is
supported, reactions are dropped one at a time while BIC does not rise (the plain search may have
added one only to imitate the site blocking).

**Measure of fit, in units of the GC noise.** Over all runs $i$ and species $s$ other than the inert
($N$ readings),
$$\chi^2 = \sum_{i}\sum_{s \ne I}\left(\frac{\hat y_{is} - y_{is}}{\sigma_{is}}\right)^2, \qquad
  \sigma_{is} = 0.01\,y_{is} + 0.0005,$$
reported as the typical miss $\sqrt{\chi^2/N}$ "times the GC noise". A network **fits within the GC
noise** when $\chi^2 \le \chi^2_{0.99}(N - p)$, no species has a typical miss above 2, and every
product the GC sees is made. The virtual GC draws the relative and the absolute part of its noise
independently, so the real scatter is up to 1.4 times smaller than $\sigma$: a right network
misses by about 0.5–0.8, and the test is lenient by about that factor. Networks are compared by
$$\text{BIC} = \chi^2 + p\ln N,$$
with evidence weights $w_k \propto \exp(-\Delta\text{BIC}_k/2)$. Below $\Delta\text{BIC} = 6$
(`INDISTINGUISHABLE`) two networks are reported as "the data cannot tell these apart", never as a
decision; from $\Delta\text{BIC} = 10$ (`CLEAR_GAP`) "no other network comes close" is said
(Kass and Raftery: below 6 is not strong evidence, above 10 very strong). A search step is
accepted only when it lowers BIC by more than 2.

**The library, enumerated from the atom balance** (`reaction_library`). Species: A and B (fed),
C = AB, D = A rearranged, E = AB₂, I (inert). Every multiset of 1–3 molecules is a possible
"side"; two sides with the same atoms, no species in common and coprime coefficients form one
entry if, in at least one direction, at most 3 molecules react and 2 are formed. Each entry has up
to three variants: one-way forward, one-way reverse, reversible. Thermodynamic pruning removes
what cannot be seen in the operating box: with $K_x = K(T)\,P^{-\Delta n}$ ($\Delta n$ = change in
the number of molecules), a direction with $K_x > 100$ at every temperature (450–850 K) and pressure
(1–100 bar) of the box is complete, and only that one-way variant is kept. Result: 12 reactions,
22 forms (30 without pruning). The search further uses only reactions whose products are fed or
seen by the GC.

| Species (placeholders) | A | B | C | D | E |
|---|---|---|---|---|---|
| composition | A | B | AB | A | AB₂ |
| formation enthalpy, kJ/mol | 0 | 0 | −120 | +50 | −180 |
| entropy, J/mol/K | 300 | 200 | 330 | 460 | 510 |

The table reproduces the complex network (A + B ⇌ C: −120 kJ/mol, −170 J/mol/K; C + B → E:
−60 kJ/mol; A → D: +50 kJ/mol) and makes its two one-way steps complete over the whole box.
**Caveat**: it was fitted to the complex example's answer, so the pruning removes exactly the other
forms of those two steps (D → A, D ⇌ A, E → B + C, B + C ⇌ E): it encodes the directions of that
answer. Making A → D complete also needs $S_D - S_A = 160$ J/mol/K, more than a rearrangement
would have. With the unpruned library (`reaction_library(thermo_prune=False)`) the search still
finds the same reactions on the example files, but cannot tell those two one-way steps from their
two-way forms (which predict the same inside the box) and says so. With real species the table
must come from measured or tabulated thermodynamics.

**The search** (`discover`, from no reactions or from a given network to repair). Each step tries
every single change:

- **add** a library reaction whose reactants are fed or already made: while a product the GC sees
  is made by no reaction, only reactions that make one; otherwise only reactions that touch a
  species still missed by more than the GC noise allows; none once the network has 4 reactions;
- **remove** a reaction (if the others can still run from the feed);
- **change one reaction order** to another of 0, 1, 2 (once every product seen is made);
- **replace**, only when none of the above helps: a reaction by another variant of itself, or by
  another library reaction that makes the same product.

All changes of a step are screened together (4 iterations of a batched Levenberg–Marquardt fit),
the best 3 are fitted to convergence, and the best is taken if it lowers BIC by more than 2. When
the reactions are settled every reactant order is tried at 0, 1 and 2; if an order changes, the
reactions are checked again (at most twice). The search changes one thing at a time, so it is not
guaranteed to find the best network there is: when its result does not fit within the GC noise it
starts again from the next-best first reaction, up to 3 times, and from the second new start on
also tries every reaction it may add with one of its orders changed. `one_step_revision` (8.5) is
one such step on its own. `rank_candidates` fits exactly the networks given and ranks them by BIC.

**The discriminating run** (`discriminating_run`). For the two best fitted networks, with
predicted mole fractions $\hat y^{(1)}_s(x)$, $\hat y^{(2)}_s(x)$ at a set point $x$,
$$S(x) = \sum_{s \ne I}\frac{\big(\hat y^{(1)}_s(x) - \hat y^{(2)}_s(x)\big)^2}{\sigma_s^2 + v^{(1)}_s(x) + v^{(2)}_s(x)},$$
with $\sigma_s$ the GC noise at the mean of the two predictions and $v^{(k)}_s$ the variance of
network $k$'s prediction over draws from its fitted parameters' Laplace covariance. $S$ is first
maximised with $v = 0$ over 2048 Sobol points of the box (or of the one-step box around the last
run) with two rounds of zooming in; the best points are then scored with the uncertainty, and the
run proposed is the best by that score. Set points are rounded to what can be set (0.5 bar,
0.5 mL/min, 1 °C) with enough total flow for the GC; the catalyst mass is not a knob. Reported:
the separation $\sqrt{S}$ without and with the uncertainty, and $S$ with it as about what one such
run adds to the BIC difference. Example (`complex_noisy.csv`, the true network against `close`):
205 times the GC noise apart at the best-fit values, about 67 with the uncertainty, about 4473 of
BIC for one free run; within one step of the last run 122, 44 and 1938.

Candidate networks used in the examples and in the v7 benchmark (`CANDIDATES`): `truth` (A + B ⇌ C,
C + B → E, A → D), `close` (the same with C + B → E zero order in B), `similar` (A + B → C,
A + 2 B → E with $r = k P_A P_B$, A → D: another pathway, no equilibrium) and `far` (A + B → C zero
order in B, C → D + B: two reactions, nothing makes E). Results: README section 3.9 and
`examples/discovery/WORKED_EXAMPLE.md`.

### 8.7 The profit vs selectivity trade-off (`smart/pareto.py`)

Optional; it changes nothing in the advisor and draws no random numbers from it. For settings $x$
in the allowed box (safety ranges, fixed knobs, flow minimum), the advisor's own predicted profit
$\hat y(x)$ (physics plus correction, as it plans with; range $\pm 1.64$ predictive sd, the 90 %
range of the web page) and the selectivity $S_C(x)$ of the best-estimate chemistry (of all the A
that reacted, the share that became C; range from the plausible chemistries).

**Pareto front.** A setting is dropped when another has
$\hat y(x') \ge \hat y(x)$ and $S_C(x') \ge S_C(x)$, one of them strictly. What is left runs from
the most profitable setting to the most selective one that still makes a profit; along it profit
falls as selectivity rises. It is found by search with the best-estimate chemistry: 1024 Sobol
candidates plus the runs done and points around the best run; the non-dominated ones; then 8
refinement rounds that try knob changes of $0.08 \times 0.75^r$ (scaled units, round $r$) around
points spread along the curve and half-way points between neighbours. The curve is cut at zero
predicted profit, thinned to about 12 points evenly spread along its length (both axes scaled to
0–1), and those are predicted again with every plausible chemistry.

**Balanced point (knee).** With $u(\cdot)$ scaling profit and selectivity along the curve to 0–1,
the balanced point maximises
$$\ell = u(\hat y) + u(S_C) - 1,$$
its height above the straight line joining the two ends; up to it a point of selectivity costs less
profit than beyond it. It is given only if $\ell \ge 0.05$; a straight or inward-bent curve has
none. A curve spanning less than 0.2 percentage point of selectivity is no trade-off (the noise of
one GC reading moves the selectivity read off it by 0.1 to 0.35 point) and comes back as one point.

**Aim for this point** (`spec_for_point`). The point becomes a minimum selectivity, the largest
under which the advisor still accepts it (at least half of the plausible chemistries meet it); the
advisor's other specs are kept, and the planner then looks for the best profit at that purity
(8.2–8.3). The planner searches coarsely under such a limit and, unless the point is passed to it
as a seed (`plan(seeds=...)`, `suggest(seeds=...)`; the web page does not yet do so), may head for
settings that earn less than the point (for the balanced point of the campaign below, 14.4 per hour
on the curve, it found 10.5–13.9 over five random seeds).

Measured on one campaign (complex virtual reactor, 12 runs, v5 advisor; `test_pareto.py`): about
2 s per curve; over the points predicted to earn at least 80 % of the top profit, the settings lie
within 0.33 % (of the best possible profit) of the true curve and the predicted selectivity within
0.07 percentage point, but the predicted profit is 0.4–0.65 per hour (2–3.5 %) too high along the
curve; the selective end is the weakest part (its last point 4.6 % below the true curve).

## 9. Baselines

- **Rules of thumb (script)** (`ChemistPolicy`, a scripted policy, not a human): read conversion
  and selectivity off the GC;
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
- **Neural network alone** (`nn_only`, `NNOnlyPolicy`, v7 study): the `direct` member of 8.5 on
  its own (8 small networks, knobs with log P → measured profit), with Expected Improvement on the
  ensemble mean and the spread between its networks over plain BO's candidates within the step box.
  Like plain BO it sees one profit number per run, no physics and no GC composition. It recommends
  the tested run with the best **measured** profit, the rule of the rules-of-thumb policy and the
  random walk: the ensemble does not reproduce its own training runs (bootstrap and weight
  penalty), so recommending by its mean picked runs that had measured far worse than its best.
- **Stress test**: the v5 advisor with the over-reaction C + B → E removed from its model (E still
  appears in the GC; only the GP correction can account for the lost product). The same for v6
  and, with and without network revision, for v7 (8.5).

## 10. What is still placeholder

Kinetics, prices, GC noise, the inhibition, the step limits and the formation enthalpies and
entropies of the species table (8.6) are plausible numbers, not measured ones. The next step is a log from the Micromeritics reactor: the advisor reads it
as is (`reactor_advisor.py my_log.csv`).
