"""Shared model for the blood-inventory Markov decision process.

Assumptions taken from the assignment:
- Inventory state and order quantity are integers in {0, 1, 2, 3}.
- Demand is 0, 2, or 3 with probabilities 0.1, 0.4, and 0.5.
- Inventory available before demand is capped at 3.
- End-of-period leftover inventory costs 10 per unit.
- Each unmet unit of demand costs 4.
- Reward is the negative of total cost, so maximizing reward minimizes cost.
"""
import numpy as np

STATES = range(4)
ACTIONS = range(4)
DEMANDS = (0, 2, 3)
DEMAND_PROBS = (0.1, 0.4, 0.5)
MAX_INVENTORY = 3
HOLDING_COST = 10.0
SHORTAGE_COST = 4.0
GAMMA = 0.8

def transition_and_reward(state, action, demand):
    """Return (next_state, reward) for one demand realization."""
    available = min(state + action, MAX_INVENTORY)
    next_state = max(available - demand, 0)
    shortage = max(demand - available, 0)
    cost = HOLDING_COST * next_state + SHORTAGE_COST * shortage
    return next_state, -cost

def build_model():
    """Return P[s,a,s'] and expected one-step reward r[s,a]."""
    P = np.zeros((4, 4, 4), dtype=float)
    r = np.zeros((4, 4), dtype=float)
    for s in STATES:
        for a in ACTIONS:
            for demand, probability in zip(DEMANDS, DEMAND_PROBS):
                next_state, reward = transition_and_reward(s, a, demand)
                P[s, a, next_state] += probability
                r[s, a] += probability * reward
    return P, r

def greedy_policy(P, r, value, gamma=1.0):
    """Choose an action maximizing expected reward plus continuation value."""
    q = r + gamma * np.einsum("sak,k->sa", P, value)
    return np.argmax(q, axis=1)
