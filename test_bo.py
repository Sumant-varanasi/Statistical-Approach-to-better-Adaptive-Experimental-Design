"""Checks for the BO package.  Run:  python test_bo.py"""
import numpy as np
from scipy.optimize import approx_fprime
from bo.gp import GP
from bo.acquisition import expected_improvement, propose
from bo.nn import NNEnsemble
from bo import loop


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


rng = np.random.default_rng(0)
X = rng.random((25, 3))
y = np.sin(6 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.01 * rng.standard_normal(25)

# 1. analytic likelihood gradient matches finite differences (both kernels, ARD and not)
for kernel in ("matern52", "rbf"):
    for ard in (True, False):
        g = GP(kernel=kernel, ard=ard)
        ys = (y - y.mean()) / y.std()
        p = np.r_[np.log([0.4, 0.7, 1.3][: 3 if ard else 1]), np.log(0.9), np.log(1e-2)]
        _, ga = g._nll_and_grad(p, X, ys)
        gn = approx_fprime(p, lambda q: g._nll_and_grad(q, X, ys)[0], 1e-6)
        check(f"likelihood gradient {kernel} ard={ard}", np.allclose(ga, gn, rtol=1e-4, atol=1e-4))

# 2. predictions match scikit-learn for the same fixed hyper-parameters
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
ls, s2, sn2 = np.array([0.3, 0.5, 0.8]), 1.3, 1e-3
ours = GP(fixed=dict(ls=ls, s2=s2, noise=sn2)).fit(X, y)
sk = GaussianProcessRegressor(ConstantKernel(s2, "fixed") * Matern(ls, "fixed", nu=2.5)
                              + WhiteKernel(sn2 + 1e-9, "fixed"), normalize_y=True,
                              optimizer=None, alpha=0.0).fit(X, y)
Xs = rng.random((50, 3))
m1, s1 = ours.predict(Xs, include_noise=True)
m2, s2_ = sk.predict(Xs, return_std=True)
check("posterior mean matches scikit-learn", np.allclose(m1, m2, atol=1e-6))
check("posterior sd matches scikit-learn", np.allclose(s1, s2_, atol=1e-5))

# 3. fitted GP learns that x2 is irrelevant (long length-scale) and is accurate
gp = GP(n_restarts=3).fit(X, y)
check("ARD: irrelevant input gets the longest length-scale", np.argmax(gp.ls) == 2)
Xt = rng.random((200, 3))
yt = np.sin(6 * Xt[:, 0]) + 0.5 * Xt[:, 1] ** 2
check("fitted GP test RMSE < 0.1", np.sqrt(np.mean((gp.predict(Xt)[0] - yt) ** 2)) < 0.1)
check("fitted noise sd close to true 0.01", 0.001 < np.sqrt(gp.sn2) * gp.y_std < 0.05)

# 4. posterior samples have the right mean
S = gp.sample(Xt[:20], 4000, rng)
check("Thompson samples average to the posterior mean",
      np.allclose(S.mean(0), gp.predict(Xt[:20])[0], atol=0.03))

# 5. kriging-believer conditioning leaves the mean unchanged at the fantasy point
x_new = rng.random(3)
mu_new = gp.predict(x_new)[0][0]
g2 = gp.condition_on(x_new, mu_new)
check("kriging believer keeps mean, shrinks sd",
      abs(g2.predict(x_new)[0][0] - mu_new) < 1e-6 and g2.predict(x_new)[1][0] < gp.predict(x_new)[1][0])

# 6. EI properties
mu = np.linspace(-1, 1, 11)
ei = expected_improvement(mu, np.full(11, 0.3), best=0.0)
check("EI is non-negative and increases with the predicted mean", (ei >= 0).all() and (np.diff(ei) > 0).all())

# 7. NN ensemble fits and reports larger uncertainty far from data
nn = NNEnsemble(seed=0).fit(X[:, :1] * 0.5, y)          # data only in x in [0, 0.5]
_, sd_in = nn.predict(np.array([[0.25]]))
_, sd_out = nn.predict(np.array([[1.0]]))
check("NN ensemble is less certain outside the data", sd_out[0] > sd_in[0])


# 8. log-scaled inputs: the wrapper maps the box onto itself and keeps the model working
from bo.loop import LogInputs
lo, hi = np.array([1.0, 0.05, 0.0]), np.array([100.0, 5.0, 1.0])
w = LogInputs(GP(seed=0), (0, 1), lo, hi)
G = w.g(np.array([[0.0, 0.0, 0.3], [1.0, 1.0, 0.3], [0.0909, 0.0909, 0.3]]))
check("log warp keeps the box ends and puts 10 bar / 0.5 g mid-way", np.allclose(G[:2, :2], [[0, 0], [1, 1]])
      and np.allclose(G[2, :2], 0.5, atol=0.01) and np.allclose(G[:, 2], 0.3))
w.fit(X, y)
check("log-warped GP predicts and conditions", np.isfinite(w.predict(Xs)[0]).all()
      and np.isfinite(w.condition_on(x_new, 0.0).predict(x_new)[0]).all())


# 9. end-to-end: every strategy runs on a cheap toy problem and BO finds its optimum
class Toy:
    name, dim, noise_sd, f_opt, f_ref = "toy", 2, 0.0, 1.0, 0.0
    def true(self, x):
        return float(np.exp(-20 * np.sum((np.asarray(x) - [0.7, 0.3]) ** 2)))
    def __call__(self, x):
        f = self.true(x); return f, f

for s in loop.STRATEGIES:
    if loop.STRATEGIES[s].get("warp_in"):      # reactor-specific (P and Wcat columns), not for the toy
        continue
    h = loop.run(s, Toy(), budget=20, seed=1)
    check(f"strategy {s} runs and records 20 evaluations", len(h["y_true"]) == 20)
h = loop.run("gp_ei", Toy(), budget=25, seed=1)
check("GP+EI gets within 1% of the toy optimum in 25 evaluations", h["rec"][-1] > 0.99)
print("\nAll checks passed.")
