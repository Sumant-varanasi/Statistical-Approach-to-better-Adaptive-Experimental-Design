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

## 0. Version 2: upgraded Bayesian optimisation (branch `bo-upgrade`)

Everything below section 0 describes the original (v1) scripts, which are unchanged and
still run. Version 2 adds a reusable `bo/` package, a harder and more realistic objective,
a 20-seed benchmark with statistics, a neural-network surrogate, constrained BO, and a
next-experiment tool. Open `report/results.html` for the interactive results page.

### 0.1 What changed and why

| Change | Why |
|---|---|
| **Fitted GP** (`bo/gp.py`): Matérn-5/2 kernel, one length-scale per knob, fitted noise, all chosen by maximising the marginal likelihood (analytic gradients, restarts) | v1 fixed one length-scale at 0.3 by hand and assumed no noise. The fitted GP learns which knobs matter. Checked against scikit-learn: identical predictions for the same settings. |
| **Rank output transform** (`copula_transform` in `bo/loop.py`) | Margin values run from about −150 to +29. Without the transform the GP spends its accuracy on the huge-loss region. With it, median regret drops 46× on yield (significant). |
| **Better acquisition maximisation** (`bo/acquisition.py`) | Sobol candidates plus points near the best experiments, then L-BFGS-B polishing. v1 only scored 5,000 random points. |
| **Four acquisition functions**: EI, UCB (GP-UCB schedule), PI, Thompson sampling | The plan requires at least two. |
| **Batch mode** (kriging believer, 4 experiments per round) | For running several experiments in parallel. |
| **NN surrogate** (`bo/nn.py`): deep ensemble of 5 small networks, bootstrap resampling | The plan's GP-vs-NN comparison. Uncertainty = disagreement between networks. |
| **Constrained BO** (`cgp_ei`): second GP on the selectivity spec, score = EI × P(meets spec) | Real plants have product specs. |
| **Log-scaled model inputs** (`LogInputs`): the GP sees P and W_cat on a log scale | Both span 100× ranges; fixes some runs that got stuck at near-zero catalyst. |
| **New objective "margin"** (`bo/problem.py`) | The yield objective is too easy (broad plateau), so random search keeps up. See 0.2. |
| **Measurement noise option** | Imitates GC error; BO then recommends the point with the best *predicted* value, not the luckiest reading. |
| **Baselines**: random search, Latin hypercube, Nelder-Mead simplex, and v1 BO itself | The plan asks for conventional baselines. |

### 0.2 The two objectives

| Objective | Formula | Difficulty | True optimum (P, V_A, V_B, T, W_cat) |
|---|---|---|---|
| `yield` | Y_C = F_C,out / F_A,0 | easy: ~3 % of the box is within 5 % of the best | 0.9973 at 100 bar, 1, 33.4 mL/min, 480 K, 5 g |
| `margin` | 1.0·C made − 0.20·(A + B fed) − 0.03·feed·ln P − 3.0·W_cat, flows in mmol/h | hard: ~90 % of the box loses money | 29.34 at 100 bar, 50, 49.3 mL/min, 589 K, 0.76 g |
| `margin` + spec S_C ≥ 0.97 | same, only in-spec points count | hard, constrained | 26.82 at 100 bar, 50, 50 mL/min, 547 K, 1.58 g |

**Prices are placeholders.** Replace `PRICES` in `bo/problem.py` with agreed values. True
optima are found by differential evolution and cached in `bo/optima.json`.

### 0.3 How to run

```bash
pip install -r requirements.txt
python test_bo.py              # 29 checks of the BO package (~5 min)
python run_benchmark.py        # 4 settings x up to 14 strategies x 20 seeds, resumable (~1.5 h on 2 cores)
python analyse_benchmark.py    # tables, significance tests, figures/bench_*.png
python surrogate_study.py      # GP vs NN prediction accuracy (~15 min)
python demo_bo_1d.py           # teaching figure: BO step by step on temperature
python make_report.py          # rebuilds report/results.html from the results
python suggest_next.py --demo  # next-experiment tool (see 0.6)
```
`run_benchmark.py --seeds 5 --settings margin` for a quick run. Results go in `results/`
(git-ignored); figures in `figures/`.

**Regret**, used everywhere: how far the point a method would recommend is from the true
optimum, scaled so 0 = optimum and 1 = a typical random point.

### 0.4 Results (20 seeds, 50 experiments each, placeholder kinetics and prices)

Median final regret (lower is better), with the share of seeds that got within 1 % of the optimum:

| Method | Yield | Margin | Margin + noise | Margin + spec |
|---|---|---|---|---|
| GP + EI | **0.0002** (95 %) | 0.0129 (45 %) | 0.0354 (25 %) | 0.711 (5 %) |
| GP + EI, log P & W_cat | **0.0002** (100 %) | **0.0043** (65 %) | 0.0316 (40 %) | – |
| Constrained GP + EI | – | – | – | **0.041** (35 %) |
| GP + UCB | **0.0002** (100 %) | 0.0422 (10 %) | 0.1004 (20 %) | – |
| GP + Thompson | 0.0004 (90 %) | 0.0397 (10 %) | 0.0427 (25 %) | – |
| GP + PI | 0.0050 (75 %) | 0.1138 (10 %) | 0.1481 (20 %) | – |
| GP + EI, batch of 4 | 0.0018 (90 %) | 0.1430 (15 %) | 0.1678 (15 %) | – |
| NN ensemble + EI | **0.0002** (100 %) | 0.0141 (30 %) | 0.0365 (35 %) | – |
| BO v1 (original) | 0.0077 (60 %) | 0.0610 (0 %) | 0.0610 (0 %) | 0.314 (0 %) |
| Random search | 0.0289 (20 %) | 0.3391 (0 %) | 0.3391 (0 %) | 0.687 (0 %) |
| Latin hypercube | 0.0287 (20 %) | 0.2958 (0 %) | 0.2993 (0 %) | 0.712 (0 %) |
| Nelder-Mead simplex | 0.0306 (0 %) | 0.2212 (0 %) | 0.3188 (0 %) | – |

Full tables with spread, timing and Holm-corrected tests: `results/analysis.txt`,
`results/bench_summary.csv`, `results/bench_tests.csv`.

What holds up statistically (Holm-corrected, vs GP + EI):

- **BO beats every conventional baseline** on the yield, margin and noisy-margin settings
  (p < 0.001, A12 ≥ 0.90). With the selectivity spec, only constrained BO does (see below). On the
  margin problem random search ends a median 16.8 money units/h below the best profit;
  GP + EI ends 0.64 below it.
- **The upgrade beats our original BO on yield** (46× lower median, p < 0.001). On the margin
  problems the median is 2–5× lower but **not significant**: about a quarter of GP + EI runs
  get stuck in a second local optimum (almost no catalyst, ~675 K), while v1 is consistently
  mediocre (0 % of runs within 1 %).
- **Constrained BO is essential when there is a spec**: 0.041 vs 0.711 (p < 0.001). Plain BO
  spends its budget on out-of-spec points (only ~4 % of its experiments met the spec in a
  check run, vs ~74 % for constrained BO).
- **EI is the best acquisition function here**, significantly better than PI. Its edge over
  UCB and Thompson sampling is not significant.
- **Batch of 4 costs accuracy**: significantly worse than one-at-a-time on yield and margin,
  in exchange for a quarter of the rounds.
- **The rank output transform matters on yield** (p < 0.001); on margin it is 8× better in
  the median but not significant.
- **GP vs NN**: as BO surrogates they tie on the margin problems; on yield GP + EI is
  slightly but significantly better (p = 0.03). The NN is 4–5× slower. As pure predictors (`surrogate_study.py`) the NN ensemble is as accurate as the
  fitted GP over the whole box and more accurate near the optimum, with better-calibrated
  intervals there. Prediction accuracy alone does not decide which surrogate optimises best.
- Nelder-Mead starts from the centre of the box, so it is identical for every seed on the
  noise-free problems.

### 0.5 Files added in v2

| File | What it does |
|---|---|
| `bo/gp.py` | GP with fitted hyper-parameters, posterior sampling, fantasy conditioning |
| `bo/acquisition.py` | EI, PI, UCB, Thompson, constrained EI, and the acquisition maximiser |
| `bo/nn.py` | deep-ensemble NN surrogate |
| `bo/loop.py` | every strategy, the BO loop, output transform, log-input wrapper |
| `bo/problem.py` | objectives, noise, product spec, true optima (`bo/optima.json`) |
| `test_bo.py` | 29 checks: gradients, match with scikit-learn, ARD, sampling, every strategy runs |
| `run_benchmark.py`, `analyse_benchmark.py` | multi-seed benchmark and its statistics/figures |
| `surrogate_study.py` | GP vs NN prediction accuracy and calibration |
| `demo_bo_1d.py` | `figures/bo_1d_steps.png`, BO explained one step at a time |
| `suggest_next.py` | next-experiment recommender for a CSV of real or simulated experiments |
| `make_report.py`, `report/template.html` | builds `report/results.html` |

### 0.6 Next-experiment tool

```bash
python suggest_next.py experiments.csv --y Y_C --batch 4         # columns: P, V_A, V_B, T, Wcat, Y_C
python suggest_next.py experiments.csv --y Y_C --fix T=600       # hold a knob fixed
```
It prints which knobs the result is most sensitive to, the best experiment so far, and the
suggested batch with predicted value and 95 % range, and writes `suggestions.csv`.
Suggestions only: any physical experiment still needs expert review, a safety check and
supervisor approval.

---

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

> **Superseded by section 0.4.** These are single-seed numbers. Over 10 seeds the
> "17 vs 285 evaluations to reach 0.99" result shrinks to a median of 22 vs ~104, and at
> 0.95 random search is as fast as BO. Use the 20-seed results above.

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

Done in v2: fitted GP length-scales (one per knob), UCB and a third acquisition function,
repeated seeds, a harder objective, the NN surrogate.

Still open, in priority order:
- **Real kinetics and prices.** Everything is placeholder. Re-run sections 0.3 once agreed.
- **Stuck runs on the margin problem.** About a quarter of GP + EI runs settle in a second
  local optimum. Candidates: a trust-region method (TuRBO), a larger initial design, or a
  portfolio of acquisition functions. Test with the existing benchmark.
- **More seeds** for the margin comparisons with v1 and between acquisition functions,
  which are not significant at 20 seeds.
- **Constrained + noisy** combined, and a constraint on safety limits as well as product spec.
- **Inert flow** is still fixed at 10 mL/min; adding it as a sixth knob is a one-line change
  in `bo/problem.py` plus the twin call.
- **Uncertainty calibration in BO**: the fitted GP's 95 % intervals cover only ~75–85 % of
  points near the margin optimum (over-confident there).
