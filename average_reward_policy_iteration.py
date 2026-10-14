"""Average-reward policy iteration for the blood-inventory MDP."""
import numpy as np
from mdp_model import build_model, greedy_policy

def evaluate_average_reward_policy(P_pi, r_pi):
    """Solve rho + h(s) = r(s) + sum P(s,s')h(s'), with h(0)=0."""
    n = len(r_pi)
    # Unknowns are rho and h[1:] (h[0] is fixed at zero).
    matrix = np.zeros((n, n), dtype=float)
    rhs = r_pi.copy()
    for s in range(n):
        matrix[s, 0] = 1.0  # coefficient of average reward rho
        for j in range(1, n):
            matrix[s, j] = (1.0 if s == j else 0.0) - P_pi[s, j]
    solution = np.linalg.solve(matrix, rhs)
    rho = solution[0]
    h = np.zeros(n)
    h[1:] = solution[1:]
    return rho, h

def policy_iteration(max_iterations=10_000):
    P, r = build_model()
    policy = np.zeros(4, dtype=int)
    for _ in range(max_iterations):
        P_pi = P[np.arange(4), policy, :]
        r_pi = r[np.arange(4), policy]
        average_reward, bias = evaluate_average_reward_policy(P_pi, r_pi)
        improved = greedy_policy(P, r, bias, gamma=1.0)
        if np.array_equal(improved, policy):
            return policy, average_reward, bias
        policy = improved
    raise RuntimeError("Policy iteration did not converge.")

if __name__ == "__main__":
    policy, average_reward, bias = policy_iteration()
    print("Optimal policy (state -> order quantity):", policy.tolist())
    print(f"Average reward per period: {average_reward:.4f}")
    print("Relative value (bias) by state:", np.round(bias, 4).tolist())
