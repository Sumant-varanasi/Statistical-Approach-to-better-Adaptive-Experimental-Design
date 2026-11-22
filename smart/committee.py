"""
v6: a committee for the physics correction.

The advisor's physics model (kinetics.py) predicts the profit of a run; what it gets wrong is the
RESIDUAL  r = measured profit - physics prediction.  v5 corrects it with one Gaussian process.
v6 lets several models predict the residual and lets the data decide how much to trust each:

  member   what it says                                       good at
  -------  -------------------------------------------------  ------------------------------------
  none     "the physics is right; r is only GC noise"         a model that is already correct
  gp       Gaussian process on the knobs (the v5 correction)  few runs, smooth errors
  nn       ensemble of small neural networks on the knobs     more runs, errors that follow the
           AND the physics prediction                         physics (e.g. a missing reaction)

How much each member is trusted is decided by STACKING (Wolpert 1992; Yao et al. 2018):
every member predicts runs it was NOT trained on (leave-one-out up to 12 runs, 6-fold after),
and the weights are the ones that make the weighted mixture predict those held-out runs best
(maximum held-out log likelihood, solved by EM).  So:
  * with few runs the GP (or "none") carries the weight, because the networks cannot yet
    predict held-out runs; the networks only join from MIN_NN runs on;
  * a member that predicts badly gets a weight near zero, so adding members cannot make the
    correction worse than its best member by much;
  * the weights are reported, so it is visible which model the advisor trusts and why.

The mixture's spread = each member's own uncertainty + how much the members DISAGREE.  The
planner explores where uncertainty is high, so disagreement between the committee members
draws experiments to where the models differ: query by committee (Seung et al. 1992).

The networks see the physics prediction as an input, so they learn "how wrong is the physics
here, given what the physics says", which is the shape of the error a missing reaction or a
wrong rate law leaves; the GP sees only the knobs.  Network outputs are clipped to the range
of residuals seen (plus a margin), so they cannot invent large corrections far from the data.
"""
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from bo.gp import GP
from bo.loop import LogInputs
from .systems import LOW, HIGH

MEMBERS = ("none", "gp", "nn")
MIN_GP = 4          # runs before any correction (as in v5)
MIN_NN = 8          # runs before the networks may join the committee
LOO_UP_TO = 12      # leave-one-out up to this many runs, K-fold after
K_FOLD = 6
# pseudo-counts on the weights (a Dirichlet prior): keep EM away from exact 0 / 1 on tiny data and,
# when members predict the held-out runs equally well, prefer the simpler one (Occam)
PSEUDO = {"none": 1.0, "gp": 0.5, "nn": 0.25}


class SmoothGP(GP):
    """Same GP as the v5 correction: length-scales of at least 0.15 in scaled units."""
    LS_BOUNDS = (np.log(0.15), np.log(20.0))


def _norm_logpdf(y, mu, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


def stacking_weights(logp, pseudo=None, iters=500):
    """Weights w on the simplex maximising sum_i log sum_k w_k p_ik (held-out predictive
    densities logp, shape (n, K)) plus a Dirichlet prior with pseudo-counts `pseudo` (K,), by EM.
    The search starts from the prior, so ties are resolved in its favour."""
    n, K = logp.shape
    a = np.full(K, 0.5) if pseudo is None else np.asarray(pseudo, float)
    lp = logp - logp.max(1, keepdims=True)          # per-row shift: cancels in the responsibilities
    p = np.exp(lp)
    w = a / a.sum()
    for _ in range(iters):
        r = w * p
        r /= np.maximum(r.sum(1, keepdims=True), 1e-300)
        w_new = (r.sum(0) + a) / (n + a.sum())
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


class _GPMember:
    """The v5 residual GP (log P input); held-out predictions reuse the full-data
    hyper-parameters (the usual approximation, and it keeps the cost tiny)."""

    def __init__(self, seed, noise_bounds):
        self.model = LogInputs(SmoothGP(seed=seed, noise_bounds=noise_bounds), (0,), LOW, HIGH)

    def fit(self, X, r):
        self.model.fit(X, r)
        return self

    def predict(self, X, include_noise=False):
        return self.model.predict(X, include_noise=include_noise)

    def heldout(self, X, r, folds):
        g = self.model.m
        Z = self.model.g(X)
        mu, var = np.zeros(len(r)), np.zeros(len(r))
        for te in folds:
            tr = np.setdiff1d(np.arange(len(r)), te)
            m_tr = r[tr].mean()                       # the held-out run must not set the mean
            K = g.kfun(Z[tr], Z[tr], g.ls, g.s2) + (g.sn2 + 1e-9) * np.eye(len(tr))
            L = np.linalg.cholesky(K)
            a = cho_solve((L, True), (r[tr] - m_tr) / g.y_std)
            k = g.kfun(Z[tr], Z[te], g.ls, g.s2)
            v = solve_triangular(L, k, lower=True)
            mu[te] = m_tr + g.y_std * (k.T @ a)
            var[te] = g.y_std ** 2 * (np.maximum(g.s2 - np.sum(v * v, 0), 1e-12) + g.sn2)
        return mu, var


class BatchMLP:
    """Many small networks trained at once: M networks, one hidden tanh layer each, linear output,
    L2 penalty, each with its own sample weights (bootstrap counts, held-out runs at 0) and its
    own targets.  The losses are independent, so one L-BFGS run on all parameters trains every
    network (same model as scikit-learn's MLPRegressor with solver='lbfgs', vectorised)."""

    def __init__(self, hidden=16, alpha=0.05, max_iter=300):
        self.h, self.alpha, self.max_iter = hidden, alpha, max_iter

    def _unpack(self, p):
        M, d, h = self.M, self.d, self.h
        q = p.reshape(M, -1)
        W1 = q[:, :d * h].reshape(M, d, h)
        b1 = q[:, d * h:d * h + h]
        W2 = q[:, d * h + h:d * h + 2 * h]
        return W1, b1, W2, q[:, -1]

    def _loss(self, p, X, Y, Wt, nw):
        W1, b1, W2, b2 = self._unpack(p)
        A = np.tanh(np.einsum("nd,mdh->mnh", X, W1) + b1[:, None, :])        # (M, n, h)
        e = np.einsum("mnh,mh->mn", A, W2) + b2[:, None] - Y                  # (M, n)
        L = 0.5 * np.sum(Wt * e * e / nw[:, None]) + 0.5 * self.alpha * np.sum(
            (np.sum(W1 * W1, (1, 2)) + np.sum(W2 * W2, 1)) / nw)
        ge = Wt * e / nw[:, None]                                             # (M, n)
        gW2 = np.einsum("mnh,mn->mh", A, ge) + self.alpha * W2 / nw[:, None]
        gb2 = ge.sum(1)
        gA = ge[:, :, None] * W2[:, None, :] * (1 - A * A)                    # (M, n, h)
        gW1 = np.einsum("nd,mnh->mdh", X, gA) + self.alpha * W1 / nw[:, None, None]
        gb1 = gA.sum(1)
        g = np.concatenate([gW1.reshape(self.M, -1), gb1, gW2, gb2[:, None]], 1)
        return L, g.ravel()

    def fit(self, X, Y, Wt, rng):
        """X (n, d) shared inputs; Y (M, n) targets; Wt (M, n) sample weights."""
        self.M, n = Y.shape
        self.d = X.shape[1]
        d, h = self.d, self.h
        lim1, lim2 = np.sqrt(6 / (d + h)), np.sqrt(6 / (h + 1))
        p0 = np.concatenate([rng.uniform(-lim1, lim1, (self.M, d * h)), np.zeros((self.M, h)),
                             rng.uniform(-lim2, lim2, (self.M, h)), np.zeros((self.M, 1))], 1).ravel()
        nw = np.maximum(Wt.sum(1), 1.0)
        res = minimize(self._loss, p0, args=(X, Y, Wt, nw), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.p = res.x
        return self

    def predict(self, X):
        """(M, m): every network's prediction at the rows of X."""
        W1, b1, W2, b2 = self._unpack(self.p)
        A = np.tanh(np.einsum("nd,mdh->mnh", X, W1) + b1[:, None, :])
        return np.einsum("mnh,mh->mn", A, W2) + b2[:, None]


class _NNMember:
    """Deep ensemble of small tanh networks, each on a bootstrap resample with its own start.
    Inputs: the knobs (log P) and the physics prediction (standardised)."""

    def __init__(self, seed, n_nets=8, hidden=16, alpha=0.05, max_iter=300):
        self.rng = np.random.default_rng(seed + 17)
        self.n_nets, self.hidden, self.alpha, self.max_iter = n_nets, hidden, alpha, max_iter
        self.tf = LogInputs(None, (0,), LOW, HIGH)

    def _features(self, X, phys):
        ph = np.clip(np.asarray(phys, float), self.ph_lo, self.ph_hi)
        return np.column_stack([self.tf.g(X), (ph - self.ph_mean) / self.ph_std])

    def _train(self, F, r, train_sets):
        """One BatchMLP holding n_nets networks per training set; returns it and the per-network
        target scaling (mean, sd), each network on a bootstrap resample of its training set."""
        Y, Wt, ym, ys = [], [], [], []
        n = len(r)
        for tr in train_sets:
            m_, s_ = r[tr].mean(), (r[tr].std() if r[tr].std() > 0 else 1.0)
            for _ in range(self.n_nets):
                w = np.bincount(self.rng.choice(tr, size=len(tr)), minlength=n).astype(float)
                Y.append((r - m_) / s_)
                Wt.append(w)
                ym.append(m_)
                ys.append(s_)
        net = BatchMLP(self.hidden, self.alpha, self.max_iter).fit(F, np.array(Y), np.array(Wt), self.rng)
        return net, np.array(ym), np.array(ys)

    def _scaled(self, net, ym, ys, F):
        return np.clip(ym[:, None] + ys[:, None] * net.predict(F), self.r_lo, self.r_hi)   # (M, m)

    def fit(self, X, r, phys, noise_var):
        phys = np.asarray(phys, float)
        span = max(phys.max() - phys.min(), 1.0)
        self.ph_lo, self.ph_hi = phys.min() - 0.25 * span, phys.max() + 0.25 * span
        self.ph_mean, self.ph_std = phys.mean(), max(phys.std(), 1e-6)
        rspan = max(r.max() - r.min(), np.sqrt(noise_var))
        self.r_lo, self.r_hi = r.min() - 0.5 * rspan, r.max() + 0.5 * rspan
        self.noise_var = noise_var
        self.F = self._features(X, phys)
        self.net, self.ym, self.ys = self._train(self.F, r, [np.arange(len(r))])
        return self

    def predict(self, X, phys, include_noise=False):
        P = self._scaled(self.net, self.ym, self.ys, self._features(X, phys))
        var = P.var(0) + (self.noise_var if include_noise else 0.0)
        return P.mean(0), np.sqrt(var + 1e-12)

    def heldout(self, X, r, phys, folds):
        n = len(r)
        trains = [np.setdiff1d(np.arange(n), te) for te in folds]
        net, ym, ys = self._train(self.F, r, trains)                # every fold's networks at once
        P = self._scaled(net, ym, ys, self.F).reshape(len(folds), self.n_nets, n)
        mu, var = np.zeros(n), np.zeros(n)
        for k, te in enumerate(folds):
            mu[te] = P[k][:, te].mean(0)
            var[te] = P[k][:, te].var(0) + self.noise_var
        return mu, var


class ResidualCommittee:
    """
    c = ResidualCommittee(seed).fit(X, resid, phys_at_data, noise_var, gp_noise_bounds)
    mu, sd, parts = c.predict(Xc, phys_at_candidates)      parts: per-member means, weights, disagreement
    c.weights                                              {member: weight}
    """

    def __init__(self, seed=0, members=MEMBERS, n_nets=8):
        self.seed, self.members, self.n_nets = seed, tuple(members), n_nets

    @staticmethod
    def folds(n, rng):
        if n <= LOO_UP_TO:
            return [np.array([i]) for i in range(n)]
        perm = rng.permutation(n)
        return [perm[k::K_FOLD] for k in range(K_FOLD)]

    def fit(self, X, resid, phys, noise_var, gp_noise_bounds=(1e-3, 1.0)):
        X = np.atleast_2d(np.asarray(X, float))
        r = np.asarray(resid, float)
        phys = np.asarray(phys, float)
        n = len(r)
        self.noise_var = float(max(noise_var, 1e-6))
        rng = np.random.default_rng(self.seed + n)
        folds = self.folds(n, rng)
        self.active, self.models, cols = [], {}, []
        if "none" in self.members:
            self.active.append("none")
            cols.append(_norm_logpdf(r, 0.0, self.noise_var))
        if "gp" in self.members and n >= MIN_GP:
            m = _GPMember(self.seed, gp_noise_bounds).fit(X, r)
            mu, var = m.heldout(X, r, folds)
            self.models["gp"] = m
            self.active.append("gp")
            cols.append(_norm_logpdf(r, mu, var))
        if "nn" in self.members and n >= MIN_NN:
            m = _NNMember(self.seed, n_nets=self.n_nets).fit(X, r, phys, self.noise_var)
            mu, var = m.heldout(X, r, phys, folds)
            self.models["nn"] = m
            self.active.append("nn")
            cols.append(_norm_logpdf(r, mu, var))
        logp = np.column_stack(cols)
        w = (stacking_weights(logp, [PSEUDO[k] for k in self.active]) if len(self.active) > 1 else np.ones(1))
        self.weights = {k: float(v) for k, v in zip(self.active, w)}
        self.heldout_logp = {k: float(v) for k, v in zip(self.active, logp.mean(0))}
        return self

    def predict(self, Xc, phys, include_noise=False):
        Xc = np.atleast_2d(Xc)
        m = len(Xc)
        mus, vars_ = {}, {}
        for k in self.active:
            if k == "none":
                mu, sd = np.zeros(m), np.full(m, np.sqrt(self.noise_var) if include_noise else 0.0)
            elif k == "gp":
                mu, sd = self.models["gp"].predict(Xc, include_noise=include_noise)
            else:
                mu, sd = self.models["nn"].predict(Xc, phys, include_noise=include_noise)
            mus[k], vars_[k] = mu, sd ** 2
        w = np.array([self.weights[k] for k in self.active])
        M = np.array([mus[k] for k in self.active])
        V = np.array([vars_[k] for k in self.active])
        mu = w @ M
        within = w @ V
        disagree = w @ (M - mu) ** 2
        sd = np.sqrt(within + disagree + 1e-12)
        return mu, sd, dict(member_mu=mus, disagreement=np.sqrt(disagree), weights=dict(self.weights))

    def describe(self):
        parts = ", ".join(f"{k} {100 * v:.0f} %" for k, v in self.weights.items())
        return f"correction committee weights: {parts}"
