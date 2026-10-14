"""
Neural-network surrogate: a deep ensemble.

A single neural network gives a prediction but no uncertainty, and BO needs both.  The
standard fix (Lakshminarayanan et al. 2017, "deep ensembles") is to train several small
networks that differ only in their random start and in a bootstrap resample of the data:
    mean        = average of the members' predictions
    uncertainty = spread (standard deviation) of the members' predictions
Where the data are dense the members agree (small sd); far from data they disagree.

Same interface as the GP:  NNEnsemble().fit(X, y);  mu, sd = model.predict(Xs)
"""
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning


class NNEnsemble:
    smooth = False      # tells the acquisition maximiser not to run gradient polishing

    def __init__(self, n_members=5, hidden=(32, 32), alpha=1e-3, bootstrap=True, seed=0,
                 max_iter=400):
        self.max_iter = max_iter
        self.n_members = n_members
        self.hidden = hidden
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.asarray(y, float)
        self.y_mean, self.y_std = y.mean(), (y.std() if y.std() > 0 else 1.0)
        ys = (y - self.y_mean) / self.y_std
        self.members = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for m in range(self.n_members):
                idx = (self.rng.integers(len(X), size=len(X)) if self.bootstrap
                       else np.arange(len(X)))
                net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh",
                                   solver="lbfgs", alpha=self.alpha, max_iter=self.max_iter,
                                   random_state=int(self.rng.integers(1 << 31)))
                net.fit(X[idx], ys[idx])
                self.members.append(net)
        return self

    def _member_preds(self, Xs):
        Xs = np.atleast_2d(Xs)
        return np.array([m.predict(Xs) for m in self.members])     # (members, m)

    def predict(self, Xs):
        P = self._member_preds(Xs)
        mu = P.mean(0)
        sd = np.sqrt(P.var(0) + 1e-4)          # small floor so EI never divides by zero
        return self.y_mean + self.y_std * mu, self.y_std * sd

    def sample(self, Xs, n_samples, rng):
        """Thompson-style draw: pick a random ensemble member for each sample."""
        P = self._member_preds(Xs)
        pick = rng.integers(len(P), size=n_samples)
        return self.y_mean + self.y_std * P[pick]
