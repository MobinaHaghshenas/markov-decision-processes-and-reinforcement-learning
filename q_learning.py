"""Tabular Q-learning for the blood-inventory MDP."""
import random
import numpy as np
from mdp_model import (
    STATES, ACTIONS, DEMANDS, DEMAND_PROBS, GAMMA, transition_and_reward
)

def q_learning(
    episodes=20_000,
    steps_per_episode=20,
    alpha=0.5,
    epsilon=0.1,
    seed=42,
):
    """Learn a discounted policy using epsilon-greedy exploration."""
    random.seed(seed)
    np.random.seed(seed)
    q_table = np.zeros((4, 4), dtype=float)

    for _ in range(episodes):
        state = random.choice(list(STATES))
        for _ in range(steps_per_episode):
            if random.random() < epsilon:
                action = random.choice(list(ACTIONS))
            else:
                # Random tie-breaking avoids a systematic preference for action 0.
                best = np.flatnonzero(q_table[state] == np.max(q_table[state]))
                action = int(random.choice(best.tolist()))

            demand = random.choices(DEMANDS, weights=DEMAND_PROBS, k=1)[0]
            next_state, reward = transition_and_reward(state, action, demand)
            target = reward + GAMMA * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state

    policy = np.argmax(q_table, axis=1)
    return policy, q_table

if __name__ == "__main__":
    policy, q_table = q_learning()
    print(f"Discount factor gamma = {GAMMA}")
    print("Learned policy (state -> order quantity):", policy.tolist())
    print("Q-table:")
    print(np.round(q_table, 4))
