"""Average-reward relative value iteration for the blood-inventory MDP."""
import numpy as np
from mdp_model import build_model, greedy_policy

def value_iteration(tolerance=1e-10, max_iterations=1_000_000):
    P, r = build_model()
    bias = np.zeros(4, dtype=float)
    for iteration in range(1, max_iterations + 1):
        q = r + np.einsum("sak,k->sa", P, bias)
        updated = np.max(q, axis=1)
        # Normalize relative values to prevent drift by an additive constant.
        updated -= updated[0]
        # Span seminorm convergence check is appropriate for relative values.
        delta = updated - bias
        if np.max(delta) - np.min(delta) < tolerance:
            bias = updated
            policy = greedy_policy(P, r, bias, gamma=1.0)
            # Estimate average reward from the reference state Bellman equation.
            q0 = r[0, policy[0]] + float(P[0, policy[0], :] @ bias)
            return policy, float(q0), bias, iteration
        bias = updated
    raise RuntimeError(
        "Relative value iteration did not converge; verify model assumptions."
    )

if __name__ == "__main__":
    policy, average_reward, bias, iterations = value_iteration()
    print(f"Converged in {iterations} iterations")
    print("Optimal policy (state -> order quantity):", policy.tolist())
    print(f"Estimated average reward per period: {average_reward:.4f}")
    print("Relative value (bias) by state:", np.round(bias, 4).tolist())
