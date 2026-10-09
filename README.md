# ACOD310 — Digital reactor twin + Bayesian optimisation

Team: Sumant Varanasi & Parva Chaudhary. Supervisors: Prof. Rachit Khare, Prof. Rahul Singh.
Project plan: see `gDoc-link`. Reactor model derivation: `digital-twin-fixed-bed-reactor.md`.

The repo has two halves that are compared against each other:

| Half | What it does | Scripts |
|---|---|---|
| **Digital twin (simulation)** | Solves the packed-bed reactor equations for one experiment, and can be run thousands of times to brute-force the best operating point. | `reactor_twin.py`, `test_reactor_twin.py`, `sweeps.py`, `simulation_campaign.py` |
| **Bayesian optimisation (BO)** | Treats the twin as an expensive black box and tries to find the best operating point in as few experiments as possible. | `bo_basic.py`, `bo_5knobs.py` |
| **Comparison** | Overlays the two on one figure. | `compare.py` |

> All kinetic parameters in `reactor_twin.py` are **placeholders**. Every number and
> figure below will change once the real kinetics are agreed with Prof. Khare.

---

## 1. How to run

```bash
pip install -r requirements.txt
```

Run from inside the repo folder, in this order (each step is independent of the ones after it):

| Step | Command | Time | Produces |
|---|---|---|---|
| 1 | `python test_reactor_twin.py` | 1 s | 11 PASS/FAIL checks of the reactor model |
| 2 | `python sweeps.py` | 5 s | `figures/1_bed_profiles.png`, `2_temperature_sweep.png`, `3_yield_map_T_W.png`, `4_pressure_sweep.png` |
| 3 | `python bo_basic.py` | 3 s | console only: 1-knob BO (temperature) vs 500-point brute force |
| 4 | `python bo_5knobs.py` | 5 s | `results/bo_history.csv`, `figures/bo_progress.png` |
| 5 | `python simulation_campaign.py` | ~1–2 min | `results/simulation_history.csv`, `figures/simulation_progress.png`, `figures/twin_5param_maps.png`, `figures/sensitivity_bar.png` |
| 6 | `python compare.py` | 1 s | `figures/overlay_bo_vs_simulation.png`, `figures/evaluations_to_target.png` + summary table |

Options: `python simulation_campaign.py --n 5000` for a quicker campaign, `--seed 2` for a
different random draw, `--grid 20` for coarser heat maps. Step 5 reads step 4's CSV so it
can mark the BO best on the heat maps; step 6 needs both CSVs.

---

## 2. Symbols used everywhere

### Species and reactions

| Symbol | Meaning |
|---|---|
| A, B | reactants fed from two mass-flow controllers (MFCs) |
| C | desired product, A + B → C (reaction 1) |
| D | unwanted by-product, A → D (reaction 2, side reaction) |
| I | inert gas (dilutes the feed, does not react) |
| r₁, r₂ | rate of reaction 1 and 2, mol per gram of catalyst per second |
| ν (`NU`) | stoichiometric matrix: how many moles of each species each reaction makes (+) or consumes (−) |

### Inputs to one experiment (the 5 "knobs" plus the inert)

| Code name | Symbol | Units | Allowed range | Meaning |
|---|---|---|---|---|
| `V_A` | V̇ₐ | mL/min (standard) | 0–50 (BO uses 1–50) | MFC flow of reactant A |
| `V_B` | V̇ʙ | mL/min | 0–50 (BO uses 1–50) | MFC flow of reactant B |
| `V_I` | V̇ɪ | mL/min | 0–50 (fixed at 10 in BO) | MFC flow of inert |
| `Wcat` | W_cat | g | 0–5 (BO uses 0.05–5) | catalyst mass in the bed |
| `T` | T | K | 298–1073 (BO uses 450–850) | reactor temperature (isothermal) |
| `P` | P | bar | 1–100 | reactor pressure (set by the back-pressure regulator) |

### Internal quantities

| Code name | Symbol | Meaning |
|---|---|---|
| `F0`, `F` | F_i | molar flow of species i, mol/s; `F0` at the inlet |
| `FT`, `FT_0`, `FT_out` | F_T | total molar flow = ΣF_i (drops as A + B → C removes molecules) |
| `y_out` | y_i | mole fraction F_i / F_T at the outlet = what the GC would report |
| `PA`, `PB` | P_i | partial pressure y_i · P, bar |
| `W` | W | catalyst mass coordinate, 0 at inlet → W_cat at outlet |
| `k1_ref`, `k2_ref` | k_ref | rate constants at the reference temperature `T_REF` = 573.15 K |
| `Ea1`, `Ea2` | E_a | activation energies, J/mol (Ea2 > Ea1 so the side reaction wins at high T) |
| `theta` | θ | the four kinetic parameters together: [k₁,ref, E_a1, k₂,ref, E_a2] |
| `R` | R | gas constant 8.314 J/(mol·K) |
| `P_STD`, `T_STD` | | MFC reference conditions (1 atm, 273.15 K) used to convert mL/min → mol/s |

### Outputs of one experiment

| Code name | Symbol | Formula | Meaning |
|---|---|---|---|
| `X_A` | X_A | (F_A,0 − F_A,out) / F_A,0 | conversion: fraction of A that reacted |
| `S_C` | S_C | F_C,out / (F_C,out + F_D,out) | selectivity: fraction of product that is the one we want |
| `Y_C` | Y_C | F_C,out / F_A,0 = X_A · S_C | **yield of C — the objective every optimiser maximises** |
| `V_out_actual_mL_min` | V̇_out | F_T,out R T / P | actual outlet volumetric flow at reactor T and P |

---

## 3. `reactor_twin.py` — the digital twin

Solves the five packed-bed mole balances dF_i/dW = Σ_j ν_ij r_j from W = 0 to W = W_cat
(sections 5–13 of the model document). Isothermal, isobaric, ideal gas, no pressure drop.

| Function | What it does |
|---|---|
| `mfc_to_molar(v_ml_min)` | standard mL/min → mol/s using F = P_std V̇ / (R T_std) |
| `arrhenius(k_ref, Ea, T)` | k(T) = k_ref · exp[−E_a/R · (1/T − 1/T_ref)] |
| `rates(F, T, P, theta)` | at one point in the bed: F_T, partial pressures, then r₁ = k₁ P_A P_B and r₂ = k₂ P_A |
| `rhs(W, F, T, P, theta)` | the ODE right-hand side, ν · r, which the integrator calls repeatedly |
| `_check_inputs(...)` | raises `ValueError` if any input is outside `LIMITS` or V_A ≤ 0 |
| `simulate(V_A, V_B, V_I, Wcat, T, P, theta=None, return_profile=False)` | **one experiment.** Builds the inlet vector, integrates with `scipy.integrate.solve_ivp` (LSODA, handles stiffness), returns a dict with `F0`, `F_out`, `FT_0`, `FT_out`, `y_out`, `X_A`, `S_C`, `Y_C`, `V_out_actual_mL_min`. With `return_profile=True` it also returns the solver object so you can evaluate F(W) anywhere in the bed. |

Module constants: `LIMITS` (allowed input ranges), `DEFAULT_THETA` (placeholder kinetics),
`NU` (stoichiometry), `SPECIES` (name order A, B, C, D, I).

One simulation costs about 1 ms, so "expensive experiment" is a simulation of the real
situation, not a property of this code.

## 4. `test_reactor_twin.py` — validation

Runs 11 checks and prints PASS/FAIL. Atom balances (all A ends up as A, C or D; all B as
B or C), inert unchanged, mole fractions sum to 1, Y_C = X_A · S_C, total flow drops by
exactly the C formed, zero catalyst → zero conversion, conversion rises with catalyst, an
exact analytical solution when reaction 1 is switched off (F_A = F_A,0 · exp(−k₂ P W / F_T,0)),
and the input-range guard.

## 5. `sweeps.py` — single-knob sweeps of the twin

Base case: P = 10 bar, V_A = V_B = 20, V_I = 10 mL/min, T = 600 K, W_cat = 1 g.

| Figure | Shows |
|---|---|
| `1_bed_profiles.png` | molar flow of every species along the bed (W from 0 to 3 g) |
| `2_temperature_sweep.png` | X_A, S_C, Y_C vs T: conversion rises with T, selectivity falls, so yield has a peak |
| `3_yield_map_T_W.png` | heat map of Y_C over T and W_cat, best grid point starred |
| `4_pressure_sweep.png` | X_A, S_C, Y_C vs P |

Also prints the mean time per simulation.

---

## 6. Bayesian optimisation — the idea in one paragraph

We have a function f(x) = Y_C(x) that is expensive to evaluate (a real experiment). BO
keeps a **Gaussian process (GP)** surrogate: given the experiments done so far it predicts,
for any untested x, a mean yield **μ(x)** and an uncertainty **σ(x)**. An **acquisition
function** turns (μ, σ) into "how worthwhile is testing x next"; we use **Expected
Improvement (EI)**, which rewards points that are either predicted to be good (high μ) or
uncertain (high σ). The point with the highest EI is run on the twin, the GP is updated,
and the loop repeats.

| Symbol in code | Meaning |
|---|---|
| `X` | experiments done so far, each knob scaled to 0–1 |
| `Y` | measured yields of those experiments; `Ys` = Y standardised to mean 0, std 1 |
| `kernel(a, b)`, `k`, `K` | RBF kernel k(a,b) = exp(−‖a−b‖² / 2ℓ²); `K` = kernel between done experiments, `k` = kernel between done experiments and candidates |
| `length` / `LENGTH` (ℓ) | length-scale: how far (in scaled units) two points can be and still have similar yields. Fixed by hand, not fitted (yet) |
| `JITTER` | 1e-6 added to K's diagonal so the linear solve is numerically stable |
| `mu` (μ) | GP predicted yield at each candidate = kᵀ K⁻¹ Ys |
| `sigma` (σ) | GP predicted uncertainty = √(1 − kᵀ K⁻¹ k) |
| `z` | (μ − best) / σ, how many standard deviations above the current best |
| `ei` | EI = (μ − best)·Φ(z) + σ·φ(z); Φ, φ = normal CDF and PDF (`scipy.stats.norm`) |
| `cand` / `grid` | candidate points that EI is evaluated at (a 500-point grid in 1-D, 5000 random points in 5-D) |
| `x_next`, `y_next` | the candidate with the highest EI and the yield the twin returns for it |

### `bo_basic.py` — 1 knob (temperature only)

The simplest possible BO, kept as the teaching version. Everything else is fixed at the
base case. Starts from 3 experiments, runs 10 BO iterations, then checks against a
500-point brute-force sweep. `f(T)` is the black box, `kernel` the RBF, and the loop
body computes μ, σ, EI and picks the next T. Note it keeps re-sampling the optimum once
converged (runs 11–13 land on the same T); a stop rule on EI is the fix.

### `bo_5knobs.py` — all 5 knobs

| Function | What it does |
|---|---|
| `unscale(x)` | scaled 0–1 vector → physical P, V_A, V_B, T, W_cat using `LOW`/`HIGH` |
| `f(x)` | the black box: one `simulate` call, returns Y_C |
| `kernel(a, b, length)` | RBF kernel across all 5 knobs (one shared length-scale) |
| `gp_posterior(X, Ys, cand, length)` | μ and σ at every candidate |
| `expected_improvement(mu, sigma, best)` | EI formula |
| `run_bo(n_init, n_iter, n_cand, length, seed)` | the loop: `N_INIT` = 10 random experiments, then `N_ITER` = 40 BO-chosen ones. Returns the full history |
| `save_history(X, Y, n_init, path)` | writes `results/bo_history.csv` (run, phase, 5 knobs, Y_C, best_so_far) |
| `plot_progress(X, Y, n_init, path)` | draws `figures/bo_progress.png` |
| `main()` | runs everything and prints the best point |

Settings at the top of the file: `N_INIT`, `N_ITER`, `N_CAND`, `LENGTH`, `JITTER`, `SEED`.
The random seed is fixed so results are reproducible.

---

## 7. `simulation_campaign.py` — brute force on the twin

The "just simulate a lot" side of the comparison. Draws `--n` random points (default
20,000) uniformly from the same 5-knob box as BO, simulates all of them, and keeps the
best. Imports the problem definition (`LOW`, `HIGH`, `f`, `unscale`) from `bo_5knobs.py`
so both halves are guaranteed to search the same space.

| Function | What it does |
|---|---|
| `run_campaign(n, seed)` | n random experiments; returns X and Y |
| `save_history(X, Y, path)` | `results/simulation_history.csv`, same columns as the BO file |
| `load_bo_best()` | reads the BO CSV (if present) to mark BO's best point on the heat maps |
| `plot_progress(Y, path)` | `figures/simulation_progress.png`: best-so-far vs number of simulations, log x-axis |
| `yield_maps(x_ref, marks, grid, path)` | `figures/twin_5param_maps.png`: 5×5 "corner" figure. Diagonal = Y_C vs one knob; below the diagonal = heat map of Y_C over a pair of knobs. In every panel the knobs not shown are held at `x_ref` (the best simulated point). Orange star = brute-force best, blue circle = BO best |
| `unscale_axis(i, v)` | helper for axis labels in physical units |
| `sensitivity_oat(base, n)` | one-at-a-time swing: sweep each knob alone across its range at the base case; swing = max Y_C − min Y_C |
| `sensitivity_global(X, Y, bins)` | first-order Sobol-style index from the random campaign: bin each knob, take the mean yield per bin, variance of those means ÷ total variance = share of yield variance that knob explains on its own |
| `plot_sensitivity(swing, s_glob, path)` | `figures/sensitivity_bar.png` (reference only, not part of the comparison) |

## 8. `compare.py` — the overlay

| Function | What it does |
|---|---|
| `load_history(path)` | reads either CSV into numpy arrays |
| `evaluations_to_target(best, target)` | first run at which best-so-far ≥ target (None if never) |
| `plot_overlay(bo, sim, path)` | `figures/overlay_bo_vs_simulation.png`: both best-so-far curves on one log-x axis, with annotations for BO's best, random search at the same 50-run budget, and the brute-force best |
| `plot_targets(bo, sim, path)` | `figures/evaluations_to_target.png`: bars of how many evaluations each method needed to first reach Y_C ≥ 0.90, 0.95, 0.98, 0.99 |
| `main()` | draws both and prints the summary table and the best operating point of each method |

---

## 9. How to read the figures

- **`bo_progress.png`** — top: dot = yield of each experiment, line = best so far, shaded = the 10 random starting experiments. Bottom: the knob settings BO picked at each step (dark = near the upper bound). You can see BO settle on high P, low V_A, high V_B, moderate T.
- **`simulation_progress.png`** — same best-so-far idea for brute force; the x-axis is log so 50 and 20,000 fit on one plot. The dotted line marks the 50-run budget BO used.
- **`overlay_bo_vs_simulation.png`** — the two curves together. The point of the project in one picture: how many evaluations each approach needs to get to a given yield.
- **`evaluations_to_target.png`** — the same information as a bar chart per target yield.
- **`twin_5param_maps.png`** — what the yield landscape looks like around the optimum across all five knobs, and that the BO point (circle) sits in the same high-yield region as the brute-force point (star).
- **`sensitivity_bar.png`** — which knobs matter. Left: swing at the base case. Right: share of variance over the whole box. Reference only.

## 10. Current results (placeholder kinetics, seed 0 for BO, seed 1 for simulation)

| | BO | brute-force random simulation |
|---|---|---|
| reactor evaluations run | 50 | 20,000 |
| best Y_C after 50 evaluations | **0.9920** | 0.9860 |
| best Y_C after all evaluations | 0.9920 | 0.9964 |
| evaluations to first reach Y_C ≥ 0.90 | 6 | 4 |
| evaluations to first reach Y_C ≥ 0.95 | 13 | 8 |
| evaluations to first reach Y_C ≥ 0.98 | 17 | 17 |
| evaluations to first reach Y_C ≥ 0.99 | **17** | **285** |

Best operating point found by each (both sit on the same high-yield plateau):

| Knob | BO | random simulation |
|---|---|---|
| P (bar) | 84.2 | 95.5 |
| V_A (mL/min) | 3.2 | 4.2 |
| V_B (mL/min) | 48.5 | 36.1 |
| T (K) | 533 | 483 |
| W_cat (g) | 4.0 | 4.8 |
| Y_C | 0.9920 | 0.9964 |

Reading: both methods find the plateau quickly because it is large (about 7 % of the box
has Y_C > 0.95). The difference shows at the top: BO reaches Y_C ≥ 0.99 in 17 evaluations,
random sampling needs 285. The remaining gap between 0.992 and 0.996 is BO stopping at 50
runs while random search kept going to 20,000.

Sensitivity (reference): pressure has the largest one-at-a-time swing at the base case;
over the whole box V_B and P explain the most yield variance and W_cat the least, because
once there is enough catalyst, adding more changes little.

## 11. Known limitations / next steps

- The GP length-scale is fixed by hand and shared by all five knobs. Fit it (and one
  length-scale per knob) by maximising the marginal likelihood.
- Only EI is implemented; the plan asks for UCB and a third acquisition function.
- One seed each. Repeat BO and random search over ~20 seeds and report medians.
- With the current objective (yield per mole of A, nothing penalising excess B, pressure
  or catalyst) the optimum is a broad plateau near Y_C ≈ 1, so random search also does
  well. A space-time-yield or cost-aware objective would make the comparison sharper;
  this is the "objective function" item in section 15 of the plan.
- NN surrogate not started yet (planned for weeks 6–8).
