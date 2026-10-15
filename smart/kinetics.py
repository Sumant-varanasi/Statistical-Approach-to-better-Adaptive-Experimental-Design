"""
Learning the chemistry from the GC readings: kinetic parameter estimation.

Every experiment gives the outlet composition from the GC (a mole fraction for every
species), not just one profit number.  With the reaction network known (which reactions
happen, their stoichiometry, mass balances, Arrhenius law, the equilibrium from
thermodynamics), the only unknowns are the rate constants and activation energies theta.
They are found by Bayesian least squares (maximum a posteriori):

    minimise   sum_ij ((y_model_ij(theta) - y_GC_ij) / sigma_ij)^2  +  sum_k ((theta_k - guess_k) / prior_sd_k)^2

    sigma_ij  = 2 % of the reading + 0.002 absolute: the GC repeatability (~1 %) plus an
                allowance for what the model's physics leaves out
    guess     = a chemist's order-of-magnitude guess (systems.py), with a WIDE prior:
                ln k within +-3 (a factor of 20 either way, 1 sd), Ea within +-48 kJ/mol (1 sd)

Uncertainty in theta comes from the curvature at the optimum (Laplace approximation):
    Cov(theta) = (J^T J)^-1 * max(1, chi2 / dof)
The chi2/dof factor inflates the uncertainty when the model fits worse than the GC noise
explains, which happens when the model's physics is incomplete (it always is a little).
Draws from N(theta_hat, Cov) are plausible versions of the chemistry; the advisor plans
the next experiment by asking all of them.

Rate constants are per m2 of catalyst surface (systems.py), so a fit made with one catalyst
loading carries over to another loading or to a batch with a different BET area.
"""
import numpy as np
from scipy.optimize import least_squares
from .systems import A_SPEC

GC_REL, GC_ABS = 0.02, 0.002          # weights in the fit: GC error + allowance for model error
GC_NOISE_REL, GC_NOISE_ABS = 0.01, 0.0005   # the GC's own repeatability (used to propagate noise)
PRIOR_SD_A, PRIOR_SD_B = 3.0, 10.0     # prior sd of ln k_ref and of Ea / (R T_ref)


class KineticFit:
    """
    kf = KineticFit(system, a_spec)
    kf.fit(phys, y_gc)              phys: (n, 6) P, V_A, V_B, V_I, T, Wcat;  y_gc: (n, ns) mole fractions
    kf.theta, kf.cov                best estimate and its covariance
    kf.samples(m, rng)              m plausible kinetics, (m, p)
    kf.predict_gc(phys)             model outlet mole fractions at the best estimate
    """

    def __init__(self, system, a_spec=A_SPEC, seed=0):
        self.S = system
        self.a_spec = float(a_spec)
        self.p = 2 * system.nr
        self.guess = system.theta_guess.copy()
        self.prior_sd = np.tile([PRIOR_SD_A, PRIOR_SD_B], system.nr)
        self.theta = self.guess.copy()
        self.cov = np.diag(self.prior_sd ** 2)
        self.rng = np.random.default_rng(seed)
        self.n_fits = 0
        self.chi2_dof = np.nan

    # ------------------------------------------------------------------ model output
    def _gc(self, phys, theta, groups=None):
        P, VA, VB, VI, T, W = np.asarray(phys, float).T
        F, _ = self.S.simulate(P, VA, VB, VI, W, T, theta=theta, groups=groups, a_spec=self.a_spec)
        return F / F.sum(1, keepdims=True)

    def predict_gc(self, phys, theta=None):
        return self._gc(np.atleast_2d(phys), self.theta if theta is None else theta)

    # ------------------------------------------------------------------ residuals + Jacobian in ONE batch
    def _res_jac(self, theta, phys, y, sig, eps=1e-4):
        """Weighted residuals and their Jacobian.  The n experiments and their p perturbed
        copies are simulated in one batch, and each experiment's copies share integration
        steps (groups), so the finite differences are smooth."""
        n, p = len(phys), self.p
        TH = np.repeat(theta[None, :], p + 1, 0)
        TH[1:] += eps * np.eye(p)
        th_rows = np.repeat(TH, n, 0)                         # (p+1)*n rows, block per theta
        ph_rows = np.tile(phys, (p + 1, 1))
        grp = np.tile(np.arange(n), p + 1)
        Y = self._gc(ph_rows, th_rows, groups=grp).reshape(p + 1, n, -1)[..., :-1]   # drop inert (redundant)
        r0 = np.nan_to_num(((Y[0] - y[:, :-1]) / sig).ravel(), nan=1e3)        # failed row = terrible fit
        Jd = np.nan_to_num(((Y[1:] - Y[0][None]) / eps / sig[None]).reshape(p, -1).T, nan=0.0)
        rp = (theta - self.guess) / self.prior_sd
        return np.r_[r0, rp], np.vstack([Jd, np.diag(1.0 / self.prior_sd)])

    def fit(self, phys, y_gc, n_starts=None):
        phys, y = np.atleast_2d(np.asarray(phys, float)), np.atleast_2d(y_gc)
        sig = GC_REL * y[:, :-1] + GC_ABS
        cache = {}

        def both(th):
            key = th.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = self._res_jac(th, phys, y, sig)
            return cache[key]

        # first fit: several starts drawn from the prior; later fits: warm start + guess
        if n_starts is None:
            n_starts = 6 if self.n_fits == 0 else 1
        starts = [self.theta.copy()] if self.n_fits else [self.guess.copy()]
        for _ in range(n_starts - 1):
            starts.append(self.guess + self.prior_sd * self.rng.normal(0, 0.7, self.p))
        if self.n_fits:
            starts.append(self.guess.copy())
        best = None
        for th0 in starts:
            try:
                res = least_squares(lambda t: both(t)[0], th0, jac=lambda t: both(t)[1],
                                    method="trf", x_scale="jac", max_nfev=60, xtol=1e-6, ftol=1e-6)
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            return self
        self.theta = best.x
        r, J = both(best.x)
        n_data = r.size - self.p
        chi2 = float(np.sum(r[:n_data] ** 2))
        self.chi2_dof = chi2 / max(n_data - self.p, 1)
        H = J.T @ J
        try:
            self.cov = np.linalg.inv(H) * max(1.0, self.chi2_dof)
        except np.linalg.LinAlgError:
            self.cov = np.linalg.pinv(H) * max(1.0, self.chi2_dof)
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.n_fits += 1
        return self

    def samples(self, m, rng):
        """m plausible kinetics: the best estimate first, then draws from the Laplace posterior."""
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(self.p))
        draws = self.theta + rng.standard_normal((m - 1, self.p)) @ L.T
        lim = 5 * self.prior_sd                 # keep draws physically sensible
        draws = np.clip(draws, self.guess - lim, self.guess + lim)
        return np.vstack([self.theta, draws])

    def param_sd(self):
        return np.sqrt(np.diag(self.cov))
