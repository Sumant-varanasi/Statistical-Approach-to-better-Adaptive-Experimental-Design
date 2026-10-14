"""
Acquisition functions and how they are maximised.

All functions take the surrogate's posterior mean `mu` and standard deviation `sd` (in
objective units) and return a score: the next experiment is the point with the highest score.

    EI   Expected Improvement      E[max(f - best - xi, 0)]
    PI   Probability of Improvement P(f > best + xi)
    UCB  Upper Confidence Bound    mu + sqrt(beta_t) * sd,  beta_t grows slowly with t
    TS   Thompson Sampling         draw one function from the posterior, take its max
                                   (handled in `propose`, it is not a closed-form score)

`propose` maximises the score in two stages, which matters in 5-D:
    1. score a large space-filling candidate set (Sobol points) PLUS points scattered around
       the best experiments so far (where the optimum usually is);
    2. polish the top few candidates with a bounded quasi-Newton search (L-BFGS-B).
The v1 code only did stage 1 with 5,000 uniform random points.
"""
import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import minimize


def expected_improvement(mu, sd, best, xi=0.0):
    imp = mu - best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def probability_of_improvement(mu, sd, best, xi=0.0):
    return norm.cdf((mu - best - xi) / sd)


def ucb_beta(t, dim, delta=0.1, scale=0.2):
    """GP-UCB exploration weight (Srinivas et al. 2010), scaled down by `scale` as is
    standard in practice (Kandasamy et al. 2017):  beta_t = scale * 2 log(d t^2 pi^2 / 6 delta)."""
    return scale * 2.0 * np.log(dim * t ** 2 * np.pi ** 2 / (6 * delta))


def upper_confidence_bound(mu, sd, beta):
    return mu + np.sqrt(beta) * sd


# ---------------------------------------------------------------------------- maximiser
def _candidates(X, y, dim, rng, n_sobol=1024, n_local=512, local_sd=0.05):
    sob = qmc.Sobol(dim, scramble=True, seed=rng.integers(1 << 31)).random(n_sobol)
    top = X[np.argsort(y)[-5:]]
    loc = top[rng.integers(len(top), size=n_local)] + rng.normal(0, local_sd, (n_local, dim))
    return np.clip(np.vstack([sob, loc]), 0, 1)


def propose(model, X, y, acq, rng, t=1, best=None, xi=0.0, n_polish=4):
    """Return the next point (scaled) chosen by acquisition `acq` in {'ei','pi','ucb','ts'}.

    model  any surrogate with predict(Xs) -> (mu, sd); for 'ts' it also needs sample()
    X, y   data so far (y = observed values)
    best   incumbent value used by EI/PI (defaults to max of the model mean at the data)
    """
    dim = X.shape[1]
    C = _candidates(X, y, dim, rng)

    if acq == "ts":
        # Thompson sampling: one joint posterior draw over a candidate set, take its argmax.
        Cs = C[rng.choice(len(C), size=min(len(C), 1000), replace=False)]
        if hasattr(model, "sample"):
            f = model.sample(Cs, 1, rng)[0]
        else:
            mu, sd = model.predict(Cs)
            f = mu + sd * rng.standard_normal(len(mu))
        return Cs[np.argmax(f)]

    if best is None:
        best = np.max(model.predict(X)[0])
    beta = ucb_beta(t, dim)

    def score(Z):
        mu, sd = model.predict(np.atleast_2d(Z))
        if acq == "ei":
            return expected_improvement(mu, sd, best, xi)
        if acq == "pi":
            return probability_of_improvement(mu, sd, best, xi)
        if acq == "ucb":
            return upper_confidence_bound(mu, sd, beta)
        raise ValueError(acq)

    s = score(C)
    starts = C[np.argsort(s)[-n_polish:]]
    best_x, best_s = C[np.argmax(s)], s.max()
    if getattr(model, "smooth", True):
        for x0 in starts:
            res = minimize(lambda z: -score(z)[0], x0, method="L-BFGS-B",
                           bounds=[(0, 1)] * dim, options={"maxiter": 50})
            if -res.fun > best_s:
                best_x, best_s = np.clip(res.x, 0, 1), -res.fun
    return best_x


# ---------------------------------------------------------------------------- constraints
def probability_feasible(model_c, Z):
    """P(constraint >= 0) under the constraint GP (Gelbart et al. 2014)."""
    mu, sd = model_c.predict(np.atleast_2d(Z))
    return norm.cdf(mu / sd)


def propose_constrained(model_f, model_c, X, y, feasible, rng, n_polish=4):
    """Constrained Expected Improvement:  score(x) = EI(x) * P(x meets the spec).
    EI is measured against the best FEASIBLE experiment so far.  Until any feasible point
    has been seen, the only goal is to find one, so the score is just P(feasible)."""
    dim = X.shape[1]
    C = _candidates(X[feasible] if feasible.any() else X,
                    y[feasible] if feasible.any() else y, dim, rng)
    if feasible.any():
        best = np.max(model_f.predict(X[feasible])[0])

    def score(Z):
        Z = np.atleast_2d(Z)
        pf = probability_feasible(model_c, Z)
        if not feasible.any():
            return pf
        mu, sd = model_f.predict(Z)
        return expected_improvement(mu, sd, best) * pf

    s = score(C)
    best_x, best_s = C[np.argmax(s)], s.max()
    for x0 in C[np.argsort(s)[-n_polish:]]:
        res = minimize(lambda z: -score(z)[0], x0, method="L-BFGS-B",
                       bounds=[(0, 1)] * dim, options={"maxiter": 50})
        if -res.fun > best_s:
            best_x, best_s = np.clip(res.x, 0, 1), -res.fun
    return best_x
