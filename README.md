# Blood Inventory Optimization with Markov Decision Processes and Reinforcement Learning

A reproducible Python implementation of a blood-inventory control problem. The project models uncertain demand as a Markov Decision Process (MDP), constructs transition and reward models, and compares dynamic-programming algorithms with model-free Q-learning.

## Problem statement

A blood bank must decide how many units to order when its current inventory is limited. Demand is uncertain, leftover inventory incurs a holding cost, and unmet demand incurs a shortage penalty. The objective is to determine an ordering policy that minimizes expected cost over time.

The implementation uses these assignment parameters:

- Inventory state: 0, 1, 2, or 3 units.
- Order quantity: 0, 1, 2, or 3 units.
- Demand distribution: 0 units with probability 0.1, 2 units with probability 0.4, and 3 units with probability 0.5.
- End-of-period holding cost: 10 per leftover unit.
- Shortage penalty: 4 per unmet unit of demand.
- Discount factor for discounted methods and Q-learning: $\gamma=0.8$.

The shortage penalty is modeled per unmet unit. This is an explicit modeling assumption in the shared model.

## Mathematical formulation

### 1. State, action, and demand

Let:

- $s_t\in\mathcal{S}=\{0,1,2,3\}$ be the inventory at the beginning of period $t$.
- $a_t\in\mathcal{A}=\{0,1,2,3\}$ be the order quantity.
- $D_t\in\{0,2,3\}$ be random demand.

The demand probabilities are

$
\Pr(D_t=0)=0.1,\qquad
\Pr(D_t=2)=0.4,\qquad
\Pr(D_t=3)=0.5.
$

### 2. Inventory and transition equations

The available inventory before demand is capped at the system capacity:

$
I_t=\min(s_t+a_t,3).
$

The inventory carried into the next period is

$
s_{t+1}=\max(I_t-D_t,0).
$

Unmet demand is

$
U_t=\max(D_t-I_t,0).
$

Thus, for a particular demand realization $d$, the transition function is

$
s'=\max(\min(s+a,3)-d,0).
$

### 3. Cost and reward

The one-period cost is

$
C(s,a,d)=10s'+4\max(d-I,0),
$

where $I=\min(s+a,3)$ and $s'=\max(I-d,0)$.

The algorithms maximize reward, so cost is converted into a negative reward:

$
R(s,a,d)=-C(s,a,d).
$

Therefore, maximizing expected reward is equivalent to minimizing expected cost under the same model.

### 4. Transition probability matrix (TPM)

For every state-action pair, the transition probability to state $s'$ is

$
P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).
$

Given the three possible demand values,

$
P(s'\mid s,a)
=\sum_{d\in\{0,2,3\}}
p(d)\,\mathbf{1}\!\left[
s'=\max(\min(s+a,3)-d,0)
\right],
$

where $p(0)=0.1$, $p(2)=0.4$, $p(3)=0.5$, and $\mathbf{1}[\cdot]$ is the indicator function.

For each action $a$, the action-specific TPM is the matrix

$
P^{(a)}_{s,s'}=P(s'\mid s,a),\qquad s,s'\in\mathcal S.
$

The implementation constructs all four action-specific TPMs from the same transition function.

### 5. Reward transition model (TRM)

The expected one-step reward for state $s$ and action $a$ is

$
r(s,a)=\mathbb{E}[R(s,a,D)].
$

Equivalently,

$
r(s,a)=\sum_{d\in\{0,2,3\}}p(d)R(s,a,d).
$

For each action $a$, the action-specific expected-reward vector is

$
r^{(a)}_s=r(s,a),\qquad s\in\mathcal S.
$

The code computes these values alongside the transition probabilities. Together, the four action-specific TPMs and four reward vectors define the finite MDP.

## Solution methods

### Discounted-reward dynamic programming

The discount factor is $\gamma=0.8$. The optimal state value satisfies the Bellman optimality equation

$
V^*(s)=\max_{a\in\mathcal A}
\left[
r(s,a)+\gamma\sum_{s'\in\mathcal S}
P(s'\mid s,a)V^*(s')
\right].
$

The optimal ordering policy is

$
\pi^*(s)=\arg\max_{a\in\mathcal A}
\left[
r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')
\right].
$

Two algorithms are implemented:

- **Policy iteration:** alternates policy evaluation and policy improvement. Policy evaluation solves $(I-\gamma P_\pi)V_\pi=r_\pi$.
- **Value iteration:** repeatedly applies the Bellman optimality update until the value function converges.

### Average-reward dynamic programming

For the long-run average-reward formulation, the average reward $\rho$ and relative value (bias) function $h(s)$ satisfy

$
\rho+h(s)=\max_{a\in\mathcal A}
\left[
r(s,a)+\sum_{s'}P(s'\mid s,a)h(s')
\right].
$

The average-reward policy iteration and value iteration implementations use this formulation. Unlike the discounted formulation, this equation does not use the discount factor.

### Q-learning

Q-learning learns action values from sampled transitions rather than requiring the transition matrix as input. The update is

$
Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)+\alpha_t
\left[
r_t+\gamma\max_{a'}Q_t(s_{t+1},a')-Q_t(s_t,a_t)
\right].
$

The implementation uses $\gamma=0.8$, a base learning-rate parameter $\alpha=0.5$, and a uniformly random behavior policy to select actions during training. It records the number of visits $v(s,a)$ for each state-action pair and uses

$
\alpha_t=\frac{0.5}{v(s_t,a_t)}.
$

After training, the reported policy is greedy with respect to the learned Q-table:

$
\pi_Q(s)=\arg\max_{a\in\mathcal A}Q(s,a).
$

Actions are selected randomly while collecting training experience, but the final reported policy chooses the action with the largest learned Q-value.

## Results

The four dynamic-programming methods returned the same ordering policy in the recorded run:

$
\boxed{\pi(s)=[2,1,0,0],\quad s=0,1,2,3.}
$

| Current inventory $s$ | Order quantity from DP |
|---:|---:|
| 0 | 2 |
| 1 | 1 |
| 2 | 0 |
| 3 | 0 |

The recorded value-iteration runs converged in 111 iterations for discounted reward and 12 iterations for average reward. The average-reward methods reported an average reward of approximately $-4$ per period for the resulting policy.

The recorded Q-learning run used 50,000 episodes and 30 steps per episode, with seed 7. Its learned greedy policy was

$
\boxed{\pi_Q(s)=[2,1,0,2].}
$

The policies agree for inventory states 0, 1, and 2. They differ at state 3. The learned Q-values for state 3 were:

| Action (order quantity) | Learned $Q(3,a)$ |
|---:|---:|
| 0 | -17.473 |
| 1 | -17.782 |
| 2 | -17.437 |
| 3 | -17.739 |

Q-learning selected order quantity 2 at state 3 because its estimated Q-value was the largest in that run. This is the output of the finite training run, not proof that Q-learning recovered the exact dynamic-programming optimum. The estimates for actions 0 and 2 at state 3 are close, so that choice should be interpreted cautiously.

## Repository structure

| File | Purpose |
|---|---|
| [MDP_Reinforcement_Learning_Analysis.ipynb](./MDP_Reinforcement_Learning_Analysis.ipynb) | Notebook for running the analysis and viewing results |
| [mdp_model.py](./mdp_model.py) | Shared state/action definitions, demand model, transition and reward functions, and TPM/TRM construction |
| [discounted_reward_policy_iteration.py](./discounted_reward_policy_iteration.py) | Discounted-reward policy iteration |
| [discounted_reward_value_iteration.py](./discounted_reward_value_iteration.py) | Discounted-reward value iteration |
| [average_reward_policy_iteration.py](./average_reward_policy_iteration.py) | Average-reward policy iteration |
| [average_reward_value_iteration.py](./average_reward_value_iteration.py) | Average-reward value iteration |
| [q_learning.py](./q_learning.py) | Tabular Q-learning with random behavior policy and visit counts |
| [requirements.txt](./requirements.txt) | Python dependencies |

## Run the project

Use Python 3.10+ in a virtual environment, Jupyter, or Google Colab.

Install dependencies:

    pip install -r requirements.txt

Run an individual algorithm from the repository directory:

    python discounted_reward_policy_iteration.py
    python discounted_reward_value_iteration.py
    python average_reward_policy_iteration.py
    python average_reward_value_iteration.py
    python q_learning.py

Alternatively, open the notebook and execute the cells in order. The notebook presents the model, algorithm outputs, and comparisons.

## Key takeaways

- The inventory problem can be represented as a finite MDP with four states and four actions.
- The demand distribution creates stochastic transitions between inventory states.
- Dynamic programming provides a model-based benchmark by using the explicitly constructed transition and reward models.
- Q-learning provides a model-free comparison based on sampled experience and a random behavior policy.
- In the recorded experiment, all four dynamic-programming methods agreed, while Q-learning matched the DP policy in three of four states. Learned policies should be evaluated rather than assumed to be optimal.

---

*This repository is an educational operations-research implementation. Results depend on the stated cost interpretation, transition assumptions, and training configuration.*
