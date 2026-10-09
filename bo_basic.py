"""
Basic Bayesian optimisation - first version.
Optimises ONE knob (temperature) with the others fixed at the base case.
Kernel settings are fixed by hand (no fitting yet).
"""
import numpy as np
from scipy.stats import norm
from reactor_twin import simulate

T_LOW, T_HIGH = 450.0, 850.0

def f(T):
    """Black box: yield at temperature T (other settings fixed)."""
    return simulate(V_A=20, V_B=20, V_I=10, Wcat=1.0, T=T, P=10)["Y_C"]

def kernel(a, b, length=0.15):
    """RBF kernel: points close together get similar yields."""
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * length ** 2))

# Start with 3 experiments (T scaled to 0-1)
X = np.array([0.1, 0.5, 0.9])
Y = np.array([f(T_LOW + x * (T_HIGH - T_LOW)) for x in X])
grid = np.linspace(0, 1, 500)          # candidate points

for i in range(10):
    # GP posterior: mean and uncertainty at every candidate
    K = kernel(X, X) + 1e-6 * np.eye(len(X))
    k = kernel(X, grid)
    mu = Y.mean() + k.T @ np.linalg.solve(K, Y - Y.mean())
    sigma = np.sqrt(np.maximum(1 - np.sum(k * np.linalg.solve(K, k), axis=0), 1e-12))

    # Expected Improvement
    z = (mu - Y.max()) / sigma
    ei = (mu - Y.max()) * norm.cdf(z) + sigma * norm.pdf(z)

    # Run the next experiment where EI is highest
    x_next = grid[np.argmax(ei)]
    y_next = f(T_LOW + x_next * (T_HIGH - T_LOW))
    X, Y = np.append(X, x_next), np.append(Y, y_next)
    print(f"run {i+4:2d}: T = {T_LOW + x_next*(T_HIGH-T_LOW):6.1f} K, "
          f"Y_C = {y_next:.4f}, best = {Y.max():.4f}")

# Check against brute force
T_all = np.linspace(T_LOW, T_HIGH, 500)
Y_all = [f(T) for T in T_all]
print(f"\nBO:          best Y_C = {Y.max():.4f} at T = {T_LOW + X[Y.argmax()]*(T_HIGH-T_LOW):.1f} K  ({len(Y)} runs)")
print(f"Brute force: best Y_C = {max(Y_all):.4f} at T = {T_all[np.argmax(Y_all)]:.1f} K  (500 runs)")
