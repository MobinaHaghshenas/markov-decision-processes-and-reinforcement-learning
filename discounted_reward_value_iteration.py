"""Discounted-reward value iteration for the blood-inventory MDP."""
import numpy as np
from mdp_model import build_model, greedy_policy, GAMMA

def value_iteration(tolerance=1e-10, max_iterations=100_000):
    P, r = build_model()
    value = np.zeros(4, dtype=float)
    for iteration in range(1, max_iterations + 1):
        q = r + GAMMA * np.einsum("sak,k->sa", P, value)
        updated = np.max(q, axis=1)
        if np.max(np.abs(updated - value)) < tolerance:
            value = updated
            return greedy_policy(P, r, value, GAMMA), value, iteration
        value = updated
    raise RuntimeError("Value iteration did not converge.")

if __name__ == "__main__":
    policy, value, iterations = value_iteration()
    print(f"Discount factor gamma = {GAMMA}")
    print(f"Converged in {iterations} iterations")
    print("Optimal policy (state -> order quantity):", policy.tolist())
    print("Discounted value by state:", np.round(value, 4).tolist())
