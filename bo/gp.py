"""
Gaussian-process surrogate with fitted hyper-parameters.

Upgrade over bo_5knobs.py, where the RBF length-scale was fixed by hand at 0.3 for every
knob and the noise was assumed to be ~zero.  Here:

    * Matern-5/2 kernel (less smooth than RBF, the usual default for BO)
    * ARD: one length-scale per knob, so the GP learns which knobs matter
    * signal variance and observation-noise variance are fitted too
    * all hyper-parameters are chosen by maximising the log marginal likelihood,
      with analytic gradients and several random restarts

Inputs are assumed scaled to [0, 1]^d.  Outputs are standardised internally and
predictions are returned in the original units.

Kernel:   k(a, b) = s2 * (1 + sqrt5 r + 5/3 r^2) * exp(-sqrt5 r),
          r^2 = sum_i ((a_i - b_i) / l_i)^2
Model:    y = f(x) + eps,  f ~ GP(c, k),  eps ~ N(0, s_n2)
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)


def _sqdist_scaled(A, B, ls):
    """Per-dimension squared distances divided by length-scale^2:  (n, m, d)."""
    D = (A[:, None, :] - B[None, :, :]) / ls
    return D * D


def matern52(A, B, ls, s2):
    """Matern-5/2 ARD kernel matrix between rows of A (n x d) and B (m x d)."""
    r = np.sqrt(np.maximum(_sqdist_scaled(A, B, ls).sum(-1), 0.0))
    return s2 * (1.0 + SQRT5 * r + 5.0 / 3.0 * r * r) * np.exp(-SQRT5 * r)


def rbf(A, B, ls, s2):
    """Squared-exponential (RBF) ARD kernel, kept for comparison with the v1 code."""
    return s2 * np.exp(-0.5 * _sqdist_scaled(A, B, ls).sum(-1))


class GP:
    """
    gp = GP().fit(X, y)
    mu, sd = gp.predict(Xs)
    samples = gp.sample(Xs, n_samples, rng)       # joint posterior draws (Thompson sampling)

    Parameters
    ----------
    kernel      'matern52' (default) or 'rbf'
    ard         one length-scale per input (True) or one shared length-scale (False)
    fixed       dict to freeze hyper-parameters instead of fitting them, e.g.
                {'ls': 0.3, 's2': 1.0, 'noise': 1e-6} reproduces the v1 GP
    noise_bounds  (min, max) of the noise variance in standardised units
    n_restarts  random restarts of the likelihood optimisation (plus a warm start)
    """

    # bounds on log hyper-parameters, in standardised-output / scaled-input units
    LS_BOUNDS = (np.log(0.02), np.log(20.0))
    S2_BOUNDS = (np.log(0.05), np.log(20.0))

    def __init__(self, kernel="matern52", ard=True, fixed=None,
                 noise_bounds=(1e-6, 0.5), n_restarts=3, seed=0):
        self.kernel_name = kernel
        self.kfun = matern52 if kernel == "matern52" else rbf
        self.ard = ard
        self.fixed = fixed
        self.noise_bounds = (np.log(noise_bounds[0]), np.log(noise_bounds[1]))
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self._warm = None          # last optimum, reused as a starting point next fit

    # ------------------------------------------------------------------ parameters
    def _unpack(self, p, d):
        nl = d if self.ard else 1
        ls = np.exp(p[:nl]) * np.ones(d)
        return ls, np.exp(p[nl]), np.exp(p[nl + 1])

    def _bounds(self, d):
        nl = d if self.ard else 1
        return [self.LS_BOUNDS] * nl + [self.S2_BOUNDS, self.noise_bounds]

    # ------------------------------------------------------------------ likelihood
    def _nll_and_grad(self, p, X, y):
        """Negative log marginal likelihood and its gradient w.r.t. the log parameters."""
        n, d = X.shape
        ls, s2, sn2 = self._unpack(p, d)
        D2 = _sqdist_scaled(X, X, ls)                  # (n, n, d)
        r = np.sqrt(np.maximum(D2.sum(-1), 0.0))
        if self.kernel_name == "matern52":
            e = np.exp(-SQRT5 * r)
            Kf = s2 * (1 + SQRT5 * r + 5.0 / 3.0 * r * r) * e
            dK_common = s2 * 5.0 / 3.0 * (1 + SQRT5 * r) * e    # dK/dlog(l_i) = this * D2_i
        else:
            Kf = s2 * np.exp(-0.5 * r * r)
            dK_common = Kf                                       # dK/dlog(l_i) = Kf * D2_i
        K = Kf + (sn2 + 1e-9) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(p)
        alpha = cho_solve((L, True), y)
        nll = 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)

        Kinv = cho_solve((L, True), np.eye(n))
        W = Kinv - np.outer(alpha, alpha)              # dNLL/dK = 0.5 * W
        grads = []
        if self.ard:
            for i in range(d):
                grads.append(0.5 * np.sum(W * dK_common * D2[:, :, i]))
        else:
            grads.append(0.5 * np.sum(W * dK_common * D2.sum(-1)))
        grads.append(0.5 * np.sum(W * Kf))              # d/dlog s2
        grads.append(0.5 * np.trace(W) * sn2)           # d/dlog sn2
        return nll, np.array(grads)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        self.X = X
        self.y_mean = y.mean()
        self.y_std = y.std() if y.std() > 0 else 1.0
        ys = (y - self.y_mean) / self.y_std
        self.ys = ys

        if self.fixed is not None:
            ls = np.asarray(self.fixed.get("ls", 0.3), float) * np.ones(d)
            self.ls, self.s2, self.sn2 = ls, self.fixed.get("s2", 1.0), self.fixed.get("noise", 1e-6)
        else:
            bounds = self._bounds(d)
            lo, hi = np.array(bounds).T
            starts = []
            if self._warm is not None and len(self._warm) == len(lo):
                starts.append(self._warm)
            nl = d if self.ard else 1
            starts.append(np.r_[np.full(nl, np.log(0.3)), 0.0, np.log(1e-3)])
            for _ in range(self.n_restarts):
                starts.append(lo + self.rng.random(len(lo)) * (hi - lo))
            best = None
            for p0 in starts:
                res = minimize(self._nll_and_grad, np.clip(p0, lo, hi), args=(X, ys),
                               jac=True, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 200})
                if best is None or res.fun < best.fun:
                    best = res
            self._warm = best.x
            self.nll = best.fun
            self.ls, self.s2, self.sn2 = self._unpack(best.x, d)

        K = self.kfun(X, X, self.ls, self.s2) + (self.sn2 + 1e-9) * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = cho_solve((self.L, True), ys)
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, Xs, include_noise=False):
        """Posterior mean and standard deviation of f (or of y if include_noise) at Xs."""
        Xs = np.atleast_2d(Xs)
        k = self.kfun(self.X, Xs, self.ls, self.s2)             # (n, m)
        mu = k.T @ self.alpha
        v = solve_triangular(self.L, k, lower=True)
        var = self.s2 - np.sum(v * v, axis=0)
        if include_noise:
            var = var + self.sn2
        sd = np.sqrt(np.maximum(var, 1e-12))
        return self.y_mean + self.y_std * mu, self.y_std * sd

    def sample(self, Xs, n_samples, rng):
        """Joint posterior samples of f at the rows of Xs: (n_samples, m)."""
        Xs = np.atleast_2d(Xs)
        k = self.kfun(self.X, Xs, self.ls, self.s2)
        mu = k.T @ self.alpha
        v = solve_triangular(self.L, k, lower=True)
        C = self.kfun(Xs, Xs, self.ls, self.s2) - v.T @ v
        C[np.diag_indices_from(C)] += 1e-8
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, U = np.linalg.eigh(C)
            Lc = U * np.sqrt(np.maximum(w, 0))
        z = rng.standard_normal((Lc.shape[1], n_samples))
        f = mu[:, None] + Lc @ z
        return (self.y_mean + self.y_std * f).T

    def condition_on(self, x_new, y_new):
        """Return a copy with an extra (fantasy) observation and the SAME hyper-parameters.
        Used by the kriging-believer batch strategy."""
        g = GP.__new__(GP)
        g.__dict__.update(self.__dict__)
        X = np.vstack([self.X, np.atleast_2d(x_new)])
        ys = np.r_[self.ys, (y_new - self.y_mean) / self.y_std]
        K = self.kfun(X, X, self.ls, self.s2) + (self.sn2 + 1e-9) * np.eye(len(X))
        g.X, g.ys = X, ys
        g.L = np.linalg.cholesky(K)
        g.alpha = cho_solve((g.L, True), ys)
        return g

    def describe(self, names=None):
        names = names or [f"x{i}" for i in range(len(self.ls))]
        ls = ", ".join(f"{n}={l:.2f}" for n, l in zip(names, self.ls))
        return (f"length-scales [{ls}]  signal var={self.s2:.2f}  "
                f"noise sd={np.sqrt(self.sn2) * self.y_std:.3g} (output units)")
