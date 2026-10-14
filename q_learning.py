
"""Tabular Q-learning for the blood-inventory MDP.

The learning rate for each state-action pair is alpha / visits(s, a).
"""

import random
import numpy as np

from mdp_model import (
    STATES,
    ACTIONS,
    DEMANDS,
    DEMAND_PROBS,
    GAMMA,
    transition_and_reward,
)


def q_learning(
    episodes=50_000,
    steps_per_episode=30,
    alpha=0.5,
    seed=42,
    return_visits=False,
):
   
    if episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("episodes and steps_per_episode must be positive.")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")

    rng = random.Random(seed)

    states = list(STATES)
    actions = list(ACTIONS)

    n_states = max(states) + 1
    n_actions = max(actions) + 1

    q_table = np.zeros((n_states, n_actions), dtype=float)
    visits = np.zeros((n_states, n_actions), dtype=int)

    for _ in range(episodes):
        # The initial inventory state is sampled randomly.
        state = rng.choice(states)

        for _ in range(steps_per_episode):
            # Assignment requirement: choose an action uniformly at random.
            action = rng.choice(actions)

            # Simulate the demand and observe the next state and reward.
            demand = rng.choices(
                DEMANDS,
                weights=DEMAND_PROBS,
                k=1,
            )[0]

            next_state, reward = transition_and_reward(
                state, action, demand
            )

            # Count observations of this state-action pair.
            visits[state, action] += 1

            # Visit-count-adjusted learning rate.
            learning_rate = alpha / visits[state, action]

            # Standard discounted Q-learning target.
            target = (
                reward
                + GAMMA * np.max(q_table[next_state])
            )

            # Temporal-difference update.
            q_table[state, action] += learning_rate * (
                target - q_table[state, action]
            )

            state = next_state

    # Extract a greedy policy; break ties randomly.
    policy = np.zeros(n_states, dtype=int)

    for state in states:
        best_actions = np.flatnonzero(
            np.isclose(q_table[state], np.max(q_table[state]))
        )
        policy[state] = int(rng.choice(best_actions.tolist()))

    if return_visits:
        return policy, q_table, visits

    return policy, q_table


if __name__ == "__main__":
    policy, q_table, visits = q_learning(return_visits=True)

    print(f"Discount factor gamma = {GAMMA}")
    print("Base learning rate alpha = 0.5")
    print("Behavior policy = uniformly random")
    print("Learned greedy policy (state -> order quantity):")
    print(policy.tolist())

    print("\nQ-table:")
    print(np.round(q_table, 4))

    print("\nState-action visit counts:")
    print(visits)

    print("\nAll state-action pairs visited:")
    print(bool(np.all(visits > 0)))
