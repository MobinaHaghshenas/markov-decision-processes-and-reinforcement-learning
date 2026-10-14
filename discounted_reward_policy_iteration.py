"""Discounted-reward policy iteration for the blood-inventory MDP."""
import numpy as np
from mdp_model import build_model, greedy_policy, GAMMA

def policy_iteration(tolerance=1e-10, max_iterations=10_000):
    P, r = build_model()
    policy = np.zeros(4, dtype=int)
    for _ in range(max_iterations):
        # Policy evaluation: (I - gamma P_pi) V = r_pi
        P_pi = P[np.arange(4), policy, :]
        r_pi = r[np.arange(4), policy]
        value = np.linalg.solve(np.eye(4) - GAMMA * P_pi, r_pi)
        improved = greedy_policy(P, r, value, GAMMA)
        if np.array_equal(improved, policy):
            return policy, value
        policy = improved
    raise RuntimeError("Policy iteration did not converge.")

if __name__ == "__main__":
    policy, value = policy_iteration()
    print(f"Discount factor gamma = {GAMMA}")
    print("Optimal policy (state -> order quantity):", policy.tolist())
    print("Discounted value by state:", np.round(value, 4).tolist())
