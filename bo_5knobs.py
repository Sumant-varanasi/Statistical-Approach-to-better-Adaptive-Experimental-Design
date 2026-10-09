"""
Basic Bayesian optimisation - second version: all 5 knobs.
Same structure as bo_basic.py. Kernel settings still fixed by hand.
Knobs: P (bar), V_A (mL/min), V_B (mL/min), T (K), Wcat (g). Inert fixed at 10 mL/min.
"""
import numpy as np
from scipy.stats import norm
from reactor_twin import simulate

np.random.seed(0)
NAMES = ["P (bar)", "V_A (mL/min)", "V_B (mL/min)", "T (K)", "Wcat (g)"]
LOW  = np.array([1.0,   1.0,  1.0,  450.0, 0.05])
HIGH = np.array([100.0, 50.0, 50.0, 850.0, 5.0])

def f(x):
    """Black box: yield for one experiment x (5 numbers, each scaled 0-1)."""
    P, VA, VB, T, W = LOW + x * (HIGH - LOW)
    return simulate(V_A=VA, V_B=VB, V_I=10, Wcat=W, T=T, P=P)["Y_C"]

def kernel(a, b, length=0.3):
    """RBF kernel: distance is measured across all 5 knobs."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * length ** 2))

# Start with 10 random experiments
X = np.random.rand(10, 5)
Y = np.array([f(x) for x in X])

for i in range(40):
    # Scale Y to mean 0, std 1 so the kernel's uncertainty level of 1 makes sense
    Ys = (Y - Y.mean()) / Y.std()

    # 5000 random candidate experiments (a full grid is impossible in 5-D)
    cand = np.random.rand(5000, 5)

    # GP posterior: mean and uncertainty at every candidate
    K = kernel(X, X) + 1e-6 * np.eye(len(X))
    k = kernel(X, cand)
    mu = k.T @ np.linalg.solve(K, Ys)
    sigma = np.sqrt(np.maximum(1 - np.sum(k * np.linalg.solve(K, k), axis=0), 1e-12))

    # Expected Improvement
    z = (mu - Ys.max()) / sigma
    ei = (mu - Ys.max()) * norm.cdf(z) + sigma * norm.pdf(z)

    # Run the next experiment where EI is highest
    x_next = cand[np.argmax(ei)]
    y_next = f(x_next)
    X, Y = np.vstack([X, x_next]), np.append(Y, y_next)
    print(f"run {i+11:2d}: Y_C = {y_next:.4f}, best = {Y.max():.4f}")

# Results
best_x = LOW + X[Y.argmax()] * (HIGH - LOW)
print(f"\nBO best Y_C = {Y.max():.4f} after {len(Y)} runs, at:")
for n, v in zip(NAMES, best_x):
    print(f"   {n:14s} {v:8.2f}")

# Check: 20,000 random experiments as a rough brute-force reference
Xr = np.random.rand(20000, 5)
Yr = np.array([f(x) for x in Xr])
print(f"\nBrute force (20,000 random runs): best Y_C = {Yr.max():.4f}")
print(f"Random search with same budget ({len(Y)} runs): best Y_C = {Yr[:len(Y)].max():.4f}")
