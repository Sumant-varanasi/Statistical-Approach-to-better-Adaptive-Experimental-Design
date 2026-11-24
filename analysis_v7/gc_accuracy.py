"""
How well does each approach predict the GC composition of runs it has NOT seen?
  mechanism  : the reaction network (rate laws + mole balances), rate constants fitted on the
               training runs only (smart/discovery.py)
  NN alone   : a neural-network ensemble from the five knobs to the GC mole fractions, no physics
               (scikit-learn MLP; size of the penalty chosen by cross-validation on the training runs)
Two tests per data file (30 runs each):
  interpolate : 5-fold cross-validation, runs held out at random
  extrapolate : train on the 24 coolest runs, predict the 6 hottest.  Here the network itself is
                found by the reaction finder from the 24 training runs alone (discover()), so
                nothing from the hot runs is used.
Error = root-mean-square miss of the mole fractions of A..E, in mol-% and as a multiple of the GC
noise (1 % relative + 0.0005 absolute, the advisor's noise model).
"""
import os, sys, json, time, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from smart import discovery as D

FILES = {"mystery_noisy.csv": "examples/discovery/mystery_reactions.txt",
         "complex_noisy.csv": "examples/discovery/candidates_truth.txt",
         "complex_real.csv": "examples/discovery/candidates_truth.txt"}


def sub(data, idx):
    return D.make_data(data["phys"][idx], data["gc"][idx], data["hours"][idx], data["a_spec"])


def feats(phys):
    return np.column_stack([np.log(phys[:, 0]), phys[:, 1:4], phys[:, 4]])


def nn_predict(ptr, ytr, pte, seed=0):
    Xtr, Xte = feats(ptr), feats(pte)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-12
    my, sy = ytr.mean(0), ytr.std(0) + 1e-12
    Xs, Ys = (Xtr - mx) / sx, (ytr - my) / sy

    def ens(alpha, X, Y, Xq, n=5):
        out = []
        for s in range(n):
            m = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", alpha=alpha, solver="lbfgs",
                             max_iter=3000, random_state=seed * 100 + s)
            m.fit(X, Y)
            out.append(m.predict(Xq))
        return np.mean(out, 0)

    best, best_a = np.inf, None
    for a in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
        err = []
        for tr, va in KFold(4, shuffle=True, random_state=seed).split(Xs):
            err.append(np.mean((ens(a, Xs[tr], Ys[tr], Xs[va], n=2) - Ys[va]) ** 2))
        if np.mean(err) < best:
            best, best_a = np.mean(err), a
    pred = ens(best_a, Xs, Ys, (Xte - mx) / sx) * sy + my
    pred = np.clip(pred, 0, 1)
    return pred, best_a


def score(pred, true):
    p, t = pred[:, :5], true[:, :5]
    return dict(rmse_mol_pct=float(100 * np.sqrt(np.mean((p - t) ** 2))),
                x_noise=float(np.sqrt(np.mean(((p - t) / D.gc_noise(t)) ** 2))))


def main():
    out = {}
    for f, net_file in FILES.items():
        data = D.load_data(f"examples/discovery/{f}")
        net = D.read_network_file(net_file)
        n = len(data["phys"])
        res = {}
        # ---- interpolation, 5-fold CV
        Pm, Pn = np.zeros_like(data["gc"]), np.zeros_like(data["gc"])
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
            fit = D.fit_network(net, sub(data, tr))
            Pm[te] = D.predict_gc(fit, data["phys"][te])
            Pn[te], _ = nn_predict(data["phys"][tr], data["gc"][tr], data["phys"][te], seed=k)
            print(f, "fold", k, flush=True)
        res["interpolate"] = dict(mechanism=score(Pm, data["gc"]), nn=score(Pn, data["gc"]))
        # ---- extrapolation to the hottest runs, network found from the cool runs only
        order = np.argsort(data["phys"][:, 4])
        tr, te = order[:24], order[24:]
        t0 = time.time()
        disc = D.discover(sub(data, tr))
        best = disc["best"]
        fit = D.fit_network(best["reactions"], sub(data, tr))
        pm = D.predict_gc(fit, data["phys"][te])
        pn, a = nn_predict(data["phys"][tr], data["gc"][tr], data["phys"][te])
        res["extrapolate"] = dict(mechanism=score(pm, data["gc"][te]), nn=score(pn, data["gc"][te]),
                                  found=best["equations"], found_model=best.get("model"),
                                  train_T_C=[float(data["phys"][tr, 4].min() - 273.15), float(data["phys"][tr, 4].max() - 273.15)],
                                  test_T_C=[float(data["phys"][te, 4].min() - 273.15), float(data["phys"][te, 4].max() - 273.15)],
                                  seconds=time.time() - t0)
        out[f] = res
        print(f, json.dumps(res, indent=1), flush=True)
    with open("analysis_v7/gc_accuracy.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("-> analysis_v7/gc_accuracy.json")


if __name__ == "__main__":
    main()
