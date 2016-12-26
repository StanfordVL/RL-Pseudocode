# Asynchronous family
 
## Asynchronous Advantage Actor-Critic (A3C)

Asynchronous Methods for Deep Reinforcement Learning. [[arxiv '16]](http://arxiv.org/abs/1602.01783)

- RMSProp with shared stats (i.e. moving average of squared gradient), updated asynchronously and without locking.
- One can explicitly use different exploration policies in each actor-learner to maximize this diversity. Thus, we do not use a replay memory and rely on parallel actors employing different exploration policies to perform the stabilizing role undertaken by experience replay in the DQN training algorithm.

### Vanilla A3C

![A3C](img/A3C.png =75%x75%)

### Async baselines

One-step Q

![Async_Q](img/Async_1Q.png =50%x50%)

N-step Q

![Async_Q](img/Async_nQ.png =75%x75%)

## Actor-Critic with Experience Replay (ACER)

Sample Efficient Actor-Critic with Experience Replay. [[arxiv '16]](http://arxiv.org/abs/1611.01224)

![ACER_1](img/ACER_1.png =50%x50%)

![ACER_2](img/ACER_2.png =75%x75%)

# DQN family

## Original

Human-level Control through Deep Reinforcement Learning. [[Nature '14]](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)

## Double DQN

Deep Reinforcement Learning with Double Q-learning. [[arxiv '15]](http://arxiv.org/abs/1509.06461)

The max operator in standard Q-learning and DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To prevent this, we can decouple the selection from the evaluation.

- Vanilla target: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta)$
- Double DQN target: $R_{t+1} + \gamma Q(S_{t+1}, argmax_a\,Q(S_{t+1}, a; \theta); \theta')$

![Double_DQN](img/Double_DQN.png =75%x75%)

## Dueling DQN

Dueling Network Architectures for Deep Reinforcement Learning. [[arxiv '15]](http://arxiv.org/abs/1511.06581)

- Dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function.
- To address the issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action. This can be achieved by subtracting either $max(\mathcal{A})$ or average $\bar{\mathcal{A}}$ from the action value. The latter is better in practice.

Formula for the decomposition of Q-value:

$$Q(s,a;\theta,\alpha,\beta) \\ = V(s;\theta,\beta) + \\ (A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a';\theta,\alpha))$$

- $\theta$ is shared parameter for the network.
- $\alpha$ parameterizes output stream for advantage function $\mathcal{A}$. 
- $\beta$ parameterizes output stream for value function _V_. 

## Prioritized Experience Replay

Prioritized Experience Replay. [[arxiv '15]](http://arxiv.org/abs/1511.05952)

We define the probability of sampling transition _i_ as 
$$ P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} $$ where $p_i$ is the priority of each transition and exponent $\alpha$ determines how much prioritization is used. $\alpha = 0$ corresponds to uniform random sampling. 

Two variants of priority assignment:

1. Proportional to TD error: $p_i = |\delta_i| + \epsilon$ where the positive $\epsilon$ ensures transitions with zero TD error will also be revisited.
2. Rank-based: $p_i = \frac{1}{rank(i)}$ sorted with respect to TD error $|\delta_i|$. More robust. 

Tricks must be used to efficiently compute the above two priorities (i.e. does not increase by _O(N)_).

Importance-sampling weights $w_i$ must be used to correct the bias introduced by prioritized replay:

$$ w_i = (\frac{1}{N} \frac{1}{P(i)})^{-\beta} $$

and linearly anneal $\beta$ from $\beta_0$ at the beginning of training to 1. 

![PER](img/prioritized_experience_replay.png =75%x75%)

## Normalized Advantage Function (NAF)

Continuous Deep Q-Learning with Model-based Acceleration. [[arxiv '16]](http://arxiv.org/abs/1603.00748)

Formulate advantage function such that the maximum is trivial to find. 

$$ Q(x, u) = V(x) + A(x, u) $$

$$ A(x, u) = -\frac{1}{2} (u - \mu(x))^T P(x) (u - \mu(x)) $$

All the above functions _Q_, _V_, _A_, _P_ are parameterized. 

$P(x)$ is a state-dependent, positive-definite square matrix where $P(x; \theta^P) = L(x; \theta^P) L(x; \theta^P)^T$ and $L(x)$ is a lower-triangular matrix whose entries come from a linear output layer of a neural network, with the diagonal terms exponentiated.

The action that maximizes _Q_ is always $\mu(x; \theta^\mu)$.

![NAF](img/NAF.png =50%x50%)

## Bootstrapped DQN

Deep Exploration via Bootstrapped DQN. [[arxiv '16]](http://arxiv.org/abs/1602.04621)

![bootstrapped](img/Bootstrapped_DQN.png =75%x75%)
