"""
Learning the chemistry from the GC readings: kinetic parameter estimation.

Every experiment gives the outlet composition from the GC (a mole fraction for every
species), not just one profit number.  With the reaction network known (which reactions
happen, their stoichiometry, mass balances, Arrhenius law, the equilibrium from
thermodynamics), the unknowns are the kinetic parameters theta.  They are found by Bayesian
least squares (maximum a posteriori):

    minimise   sum_ij ((y_model_ij(theta) - y_GC_ij) / sigma_ij)^2  +  sum_k ((theta_k - guess_k) / prior_sd_k)^2

    sigma_ij  = 2 % of the reading + 0.002 absolute: the GC repeatability (~1 %) plus an
                allowance for what the model's physics leaves out
    guess     = a chemist's order-of-magnitude guess (systems.py), with a WIDE prior:
                ln k within +-3 (a factor of 20 either way, 1 sd), Ea within +-48 kJ/mol (1 sd)

Nested model classes are fitted and the data choose between them (v5):

    plain         ln k_ref and Ea for every reaction                          (2 n_r parameters)
    inhib         + product adsorption: every rate j / (1 + K_C(T) P_C)^m_j, m_j = sites in the
                  forward step, K_C falling with temperature at the typical adsorption heat
                  (ln K_C(T) = ln K_C + e (T_ref/T - 1), e = -dH_ads/(R T_ref) ~ 10)  (+1: ln K_C)
    inhib_T       + the adsorption heat itself fitted                             (+1: e >= 0)
    decay         + catalyst activity exp(-d t) with t the time-on-stream in hours before the
                  run (d < 0 = break-in, d > 0 = deactivation)               (+1: d)

The choice is by BIC (chi2 weighted by the GC's own noise + p ln n), so an extra parameter is
only kept when it buys a clear improvement of the fit.  The advisor then reports what the
data support ("your data show product inhibition, K_C ~ 0.03 /bar").  The model never has the
virtual reactor's reactant-adsorption term, so the advisor's physics stays imperfect, as on a
real reactor.

Uncertainty in theta comes from the curvature at the optimum (Laplace approximation):
    Cov(theta) = (J^T J)^-1 * max(1, chi2 / dof)
The chi2/dof factor inflates the uncertainty when the model fits worse than the GC noise
explains.  Draws from N(theta_hat, Cov) are plausible versions of the chemistry; the advisor
plans the next experiment by asking all of them.

Rate constants are per m2 of catalyst surface (systems.py), so a fit made with one catalyst
loading carries over to another loading or to a batch with a different BET area.
"""
import numpy as np
from scipy.optimize import least_squares
from .systems import A_SPEC, T_REF

GC_REL, GC_ABS = 0.02, 0.002          # weights in the fit: GC error + allowance for model error
GC_NOISE_REL, GC_NOISE_ABS = 0.01, 0.0005   # the GC's own repeatability (used to propagate noise)
PRIOR_SD_A, PRIOR_SD_B = 3.0, 10.0     # prior sd of ln k_ref and of Ea / (R T_ref)
PRIOR_LNK, PRIOR_SD_LNK = np.log(0.01), 2.5   # prior on ln K_C at T_ref: 0.01 /bar, a factor of 12 either way
PRIOR_E, PRIOR_SD_E = 10.0, 5.0        # prior on -dH_ads/(R T_ref): ~48 +- 24 kJ/mol
PRIOR_SD_D = 0.02 / 24.0               # prior sd of the decay rate per hour (2 % activity per day)
HOURS_PER_RUN = 24.0                   # default time-on-stream of one run when the log has no hours
# extra parameters of each model, in this order: [ln K_C, e, d]
MODELS = {"plain": [], "inhib": ["lnK"], "inhib_T": ["lnK", "e"], "decay": ["lnK", "e", "d"],
          "plain_decay": ["d"]}
MIN_RUNS = {"plain": 1, "inhib": 4, "inhib_T": 6, "decay": 8, "plain_decay": 8}
_PRIOR_MEAN = {"lnK": PRIOR_LNK, "e": PRIOR_E, "d": 0.0}
_PRIOR_SD = {"lnK": PRIOR_SD_LNK, "e": PRIOR_SD_E, "d": PRIOR_SD_D}
_LOWER = {"lnK": -np.inf, "e": 0.0, "d": -np.inf}


class KineticFit:
    """
    kf = KineticFit(system, a_spec)
    kf.fit(phys, y_gc, hours)       phys: (n, 6) P, V_A, V_B, V_I, T, Wcat in the order the runs were
                                    done;  y_gc: (n, ns) mole fractions;  hours: duration of each run
    kf.model                        'plain' | 'inhib' | 'inhib_T' | 'decay' | 'plain_decay' (chosen by the data)
    kf.theta, kf.cov                best estimate (kinetics + extras) and its covariance
    kf.theta_kin                    the kinetic part only: [ln k_1, Ea_1/(R T_ref), ln k_2, ...]
    kf.samples(m, rng)              m plausible parameter vectors, (m, p)
    kf.simulate(phys6, thetas, t)   outlet flows for parameter rows `thetas` at time-on-stream t (h)
    kf.predict_gc(phys6, theta, t)  model outlet mole fractions
    kf.describe()                   what the data say about inhibition and deactivation
    """

    def __init__(self, system, a_spec=A_SPEC, seed=0, models=("plain", "inhib", "inhib_T", "decay", "plain_decay")):
        self.S = system
        self.a_spec = float(a_spec)
        self.nr = system.nr
        self.pk = 2 * system.nr
        self.models = tuple(m for m in models if m in MODELS)
        self.guess_kin = system.theta_guess.copy()
        self.prior_sd_kin = np.tile([PRIOR_SD_A, PRIOR_SD_B], system.nr)
        self.model = "plain"
        self.theta = self.guess_kin.copy()
        self.cov = np.diag(self.prior_sd_kin ** 2)
        self.rng = np.random.default_rng(seed)
        self.n_fits = 0
        self.chi2_dof = np.nan
        self.scores = {}
        self._warm = {}
        self.t_rows = None                      # time-on-stream before each fitted run (h)
        self.hours = None
        self.has_replicates = False

    # ------------------------------------------------------------------ parameter layout
    def extras(self, model=None):
        return MODELS[self.model if model is None else model]

    def guess(self, model):
        return np.r_[self.guess_kin, [_PRIOR_MEAN[e] for e in self.extras(model)]]

    def prior_sd(self, model):
        return np.r_[self.prior_sd_kin, [_PRIOR_SD[e] for e in self.extras(model)]]

    def bounds(self, model):
        ex = self.extras(model)
        lb = np.r_[np.full(self.pk, -np.inf), [_LOWER[e] for e in ex]]
        return lb, np.full(len(lb), np.inf)

    def split(self, thetas, model=None):
        """Parameter rows (m, p) -> dict with kin (m, 2 n_r) and, when the model has them,
        lnK, e, d as (m,) arrays."""
        th = np.atleast_2d(np.asarray(thetas, float))
        out = dict(kin=th[:, :self.pk])
        for j, name in enumerate(self.extras(model)):
            if th.shape[1] > self.pk + j:
                out[name] = th[:, self.pk + j]
        return out

    @property
    def theta_kin(self):
        return self.theta[:self.pk]

    def kinetics_rows(self, thetas):
        """Just the kinetic part of parameter rows (m, 2 n_r)."""
        return np.atleast_2d(np.asarray(thetas, float))[:, :self.pk]

    # ------------------------------------------------------------------ model output
    def simulate(self, phys6, thetas, t=None, groups=None, model=None):
        """Outlet flows (n, ns) and inlet flows for operating rows phys6 (n, 6) under parameter
        rows `thetas` ((n, p) or one row for all).  t = time-on-stream (h) before each row
        (activity exp(-d t)); None = fresh catalyst."""
        phys6 = np.atleast_2d(np.asarray(phys6, float))
        n = len(phys6)
        th = np.atleast_2d(np.asarray(thetas, float))
        if th.shape[0] == 1 and n > 1:
            th = np.repeat(th, n, 0)
        parts = self.split(th, model)
        kin = parts["kin"].copy()
        P, VA, VB, VI, T, W = phys6.T
        K = None
        if "lnK" in parts:
            # adsorption weakens when hotter: ln K_C(T) = ln K_C + e (T_ref/T - 1), e > 0, so K
            # falls above T_ref.  The 'inhib' model uses the typical adsorption heat (e at its
            # prior mean); 'inhib_T' fits e.
            e = parts["e"] if "e" in parts else PRIOR_E
            K = np.exp(parts["lnK"] + e * (T_REF / T - 1.0))
        if "d" in parts and t is not None:
            tt = np.broadcast_to(np.asarray(t, float), (n,))
            kin[:, 0::2] -= (parts["d"] * tt)[:, None]        # activity exp(-d t) scales every rate constant
        return self.S.simulate(P, VA, VB, VI, W, T, theta=kin, groups=groups, a_spec=self.a_spec, inhib=K)

    def predict_gc(self, phys6, theta=None, t=None):
        F, _ = self.simulate(phys6, self.theta if theta is None else theta, t=t)
        return F / F.sum(1, keepdims=True)

    # ------------------------------------------------------------------ residuals + Jacobian in ONE batch
    def _res_jac(self, theta, phys, y, sig, t_rows, guess, psd, model, eps=1e-4):
        """Weighted residuals and their Jacobian.  The n experiments and their p perturbed
        copies are simulated in one batch, and each experiment's copies share integration
        steps (groups), so the finite differences are smooth."""
        n, p = len(phys), len(theta)
        TH = np.repeat(theta[None, :], p + 1, 0)
        TH[1:] += eps * np.eye(p)
        th_rows = np.repeat(TH, n, 0)                         # (p+1)*n rows, block per theta
        ph_rows = np.tile(phys, (p + 1, 1))
        grp = np.tile(np.arange(n), p + 1)
        F, _ = self.simulate(ph_rows, th_rows, t=np.tile(t_rows, p + 1), groups=grp, model=model)
        Y = (F / F.sum(1, keepdims=True)).reshape(p + 1, n, -1)[..., :-1]   # drop inert (redundant)
        r0 = np.nan_to_num(((Y[0] - y[:, :-1]) / sig).ravel(), nan=1e3)        # failed row = terrible fit
        Jd = np.nan_to_num(((Y[1:] - Y[0][None]) / eps / sig[None]).reshape(p, -1).T, nan=0.0)
        rp = (theta - guess) / psd
        return np.r_[r0, rp], np.vstack([Jd, np.diag(1.0 / psd)])

    def _fit_model(self, model, phys, y, sig, t_rows, starts):
        guess, psd = self.guess(model), self.prior_sd(model)
        lb, ub = self.bounds(model)
        cache = {}

        def both(th):
            key = th.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = self._res_jac(th, phys, y, sig, t_rows, guess, psd, model)
            return cache[key]

        best = None
        for th0 in starts:
            th0 = np.clip(np.asarray(th0, float), lb, ub)
            try:
                res = least_squares(lambda t: both(t)[0], th0, jac=lambda t: both(t)[1], bounds=(lb, ub),
                                    method="trf", x_scale="jac", max_nfev=60, xtol=1e-6, ftol=1e-6)
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            return None
        r, J = both(best.x)
        return best, r, J

    @staticmethod
    def replicates(phys, tol=0.02, gap=2):
        """True if two runs at least `gap` runs apart repeat the same set points (within tol of
        each knob's range): the check runs an operator does to see drift and repeatability."""
        ph = np.atleast_2d(np.asarray(phys, float))[:, :5]
        from .systems import LOW, HIGH
        z = (ph - LOW) / (HIGH - LOW)
        for i in range(len(z)):
            for j in range(i + gap, len(z)):
                if np.max(np.abs(z[i] - z[j])) <= tol:
                    return True
        return False

    @staticmethod
    def time_on_stream(hours):
        """Cumulative hours before each run from the duration of every run."""
        h = np.asarray(hours, float)
        return np.r_[0.0, np.cumsum(h)[:-1]]

    def fit(self, phys, y_gc, hours=None, n_starts=None):
        phys, y = np.atleast_2d(np.asarray(phys, float)), np.atleast_2d(y_gc)
        sig = GC_REL * y[:, :-1] + GC_ABS
        n = len(phys)
        hours = np.full(n, HOURS_PER_RUN) if hours is None else np.asarray(hours, float)
        t_rows = self.time_on_stream(hours)
        self.t_rows, self.hours = t_rows, hours
        n_data = n * (y.shape[1] - 1)
        if n_starts is None:
            n_starts = 6 if self.n_fits == 0 else 1

        fits = {}
        # the plain model: several starts the first time, the previous optimum afterwards
        starts = [self._warm["plain"]] if "plain" in self._warm else [self.guess("plain")]
        for _ in range(n_starts - 1):
            starts.append(self.guess("plain") + self.prior_sd("plain") * self.rng.normal(0, 0.7, self.pk))
        if self.n_fits:
            starts.append(self.guess("plain"))
        out = self._fit_model("plain", phys, y, sig, t_rows, starts)
        if out is None:
            return self
        fits["plain"] = out

        def extend(base_model, model):
            """Start the richer model from the simpler model's optimum, new extras at their prior mean."""
            base = fits[base_model][0].x
            have = dict(zip(self.extras(base_model), base[self.pk:]))
            x0 = np.r_[base[:self.pk], [have.get(e, _PRIOR_MEAN[e]) for e in self.extras(model)]]
            starts = [x0] + ([self._warm[model]] if model in self._warm else [])
            out = self._fit_model(model, phys, y, sig, t_rows, starts)
            if out is not None:
                fits[model] = out

        # richer models, each warm-started from the one below it in the ladder.  A change of
        # activity can only be told apart from the kinetics by REPEATING a set point later on,
        # so the decay models are only tried when the log holds such a replicate pair.
        self.has_replicates = self.replicates(phys)
        for base_model, model in (("plain", "inhib"), ("inhib", "inhib_T"), ("inhib_T", "decay"), ("plain", "plain_decay")):
            if "d" in MODELS[model] and not self.has_replicates:
                continue
            if model in self.models and n >= MIN_RUNS[model] and base_model in fits:
                extend(base_model, model)

        # BIC = chi2 + p ln n, with chi2 weighted by the GC's OWN noise (the fit weights sigma
        # carry an allowance for model error, which would under-state the evidence).  An extra
        # parameter is kept only when it buys more than ln(n_data) in chi2.
        w = (sig / (GC_NOISE_REL * y[:, :-1] + GC_NOISE_ABS)).ravel()
        self.scores = {}
        for m, (res, r, _) in fits.items():
            p = self.pk + len(MODELS[m]) + ("d" in MODELS[m])    # a drift claim must be clearly supported
            chi2 = float(np.sum((r[:n_data] * w) ** 2))
            self.scores[m] = float(chi2 + p * np.log(n_data))
        self.model = min(self.scores, key=self.scores.get)
        res, r, J = fits[self.model]
        for m, (rm, _, _) in fits.items():
            self._warm[m] = rm.x
        self.theta = res.x
        p = len(res.x)
        chi2 = float(np.sum(r[:r.size - p] ** 2))
        self.chi2_dof = chi2 / max(n_data - p, 1)
        H = J.T @ J
        try:
            self.cov = np.linalg.inv(H) * max(1.0, self.chi2_dof)
        except np.linalg.LinAlgError:
            self.cov = np.linalg.pinv(H) * max(1.0, self.chi2_dof)
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.n_fits += 1
        return self

    def samples(self, m, rng):
        """m plausible parameter vectors: the best estimate first, then draws from the Laplace
        posterior, kept inside physically sensible ranges."""
        p = len(self.theta)
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(p))
        draws = self.theta + rng.standard_normal((m - 1, p)) @ L.T
        guess, psd = self.guess(self.model), self.prior_sd(self.model)
        lim = 5 * psd
        draws = np.clip(draws, guess - lim, guess + lim)
        lb, ub = self.bounds(self.model)
        draws = np.clip(draws, lb, ub)
        return np.vstack([self.theta, draws])

    def param_sd(self):
        return np.sqrt(np.diag(self.cov))

    def describe(self, rng=None, T=None):
        """What the data say beyond the rate constants, with likely ranges.  K_C is given at
        temperature T (K; default T_ref); the activity loss per day of time-on-stream."""
        parts = self.split(self.samples(400, rng or np.random.default_rng(1)))
        out = dict(model=self.model, scores=dict(self.scores))
        if "lnK" in parts:
            lnK = parts["lnK"]
            e = parts["e"] if "e" in parts else PRIOR_E
            if T is not None:
                lnK = lnK + e * (T_REF / T - 1.0)
            K = np.exp(lnK)
            out.update(K_C=float(np.median(K)), K_C_lo=float(np.percentile(K, 10)), K_C_hi=float(np.percentile(K, 90)),
                       K_at_T=float(T if T is not None else T_REF))
            if "e" in parts:
                dH = -parts["e"] * 8.314 * T_REF / 1000.0
                out.update(dH_ads=float(np.median(dH)), dH_ads_lo=float(np.percentile(dH, 10)),
                           dH_ads_hi=float(np.percentile(dH, 90)))
        if "d" in parts:
            loss = 1.0 - np.exp(-parts["d"] * 24.0)             # per day of time-on-stream
            out.update(loss_per_day=float(np.median(loss)), loss_lo=float(np.percentile(loss, 10)),
                       loss_hi=float(np.percentile(loss, 90)),
                       t_total=float(self.t_rows[-1] + self.hours[-1]) if self.t_rows is not None else 0.0)
        return out
