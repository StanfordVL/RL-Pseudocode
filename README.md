# Asynchronous family
 
## Asynchronous Advantage Actor-Critic (A3C)

Asynchronous Methods for Deep Reinforcement Learning. [[arxiv '16]](http://arxiv.org/abs/1602.01783)

- RMSProp with shared stats (i.e. moving average of squared gradient), updated asynchronously and without locking.
- One can explicitly use different exploration policies in each actor-learner to maximize this diversity. Thus, we do not use a replay memory and rely on parallel actors employing different exploration policies to perform the stabilizing role undertaken by experience replay in the DQN training algorithm.

### Vanilla A3C

![A3C](img/A3C.png)

### Async baselines

One-step Q
<img src="img/Async_1Q.png" alt="Async_Q" width="649" height="675" />

N-step Q
![Async_Q](img/Async_nQ.png)

## Actor-Critic with Experience Replay (ACER)

Sample Efficient Actor-Critic with Experience Replay. [[arxiv '16]](http://arxiv.org/abs/1611.01224)

![ACER_1](img/ACER_1.png)

![ACER_2](img/ACER_2.png)

# DQN family

## Original

Human-level Control through Deep Reinforcement Learning. [[Nature '14]](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)

## Double DQN

Deep Reinforcement Learning with Double Q-learning. [[arxiv '15]](http://arxiv.org/abs/1509.06461)

The max operator in standard Q-learning and DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To prevent this, we can decouple the selection from the evaluation.

- Vanilla target: <img src="_gitex/tex_0803ba81f2fab1c4a4aeb8a3544d95f9.png" alt="R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta)"  height="17" />
- Double DQN target: <img src="_gitex/tex_0da09c3ac1d534875b4247dcae32f765.png" alt="R_{t+1} + \gamma Q(S_{t+1}, argmax_a\,Q(S_{t+1}, a; \theta); \theta')"  height="17" />

![Double_DQN](img/Double_DQN.png)

## Dueling DQN

Dueling Network Architectures for Deep Reinforcement Learning. [[arxiv '15]](http://arxiv.org/abs/1511.06581)

- Dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function.
- To address the issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action. This can be achieved by subtracting either <img src="_gitex/tex_92c1ebec698f9d396676d7f62b6f7860.png" alt="max(\mathcal{A})"  height="17" /> or average <img src="_gitex/tex_8d9f4d43dd58fa69dc5dc5381763f2ac.png" alt="\bar{\mathcal{A}}"  height="15" /> from the action value. The latter is better in practice.

Formula for the decomposition of Q-value:

<img src="_gitex/tex_1ba08fcb2d7574a32d4d6920c8cc0dfc.png" alt="Q(s,a;\theta,\alpha,\beta) \\ = V(s;\theta,\beta) + \\ (A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a';\theta,\alpha))"  height="51" />

- <img src="_gitex/tex_25328cdc28b5efd70b1ff684010b3840.png" alt="\theta"  height="12" /> is shared parameter for the network.
- <img src="_gitex/tex_29ad4d780791e8f06fe23e545dfac699.png" alt="\alpha"  height="7" /> parameterizes output stream for advantage function <img src="_gitex/tex_c8d5b1b8f8e1e08a507931cb4f91f3fe.png" alt="\mathcal{A}"  height="13" />. 
- <img src="_gitex/tex_d2df1bc88fc935017f9ba3c2c41ab83f.png" alt="\beta"  height="15" /> parameterizes output stream for value function _V_. 

## Prioritized Experience Replay

Prioritized Experience Replay. [[arxiv '15]](http://arxiv.org/abs/1511.05952)

We define the probability of sampling transition _i_ as 
<img src="_gitex/tex_cb83dfd89b099f0d25ddaca52c9b327e.png" alt=" P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} "  height="46" /> where <img src="_gitex/tex_4720f569b0fa044352808cf82e3c2f86.png" alt="p_i"  height="11" /> is the priority of each transition and exponent <img src="_gitex/tex_29ad4d780791e8f06fe23e545dfac699.png" alt="\alpha"  height="7" /> determines how much prioritization is used. <img src="_gitex/tex_f79b1ab1598eb601f44b86439921bb4d.png" alt="\alpha = 0"  height="12" /> corresponds to uniform random sampling. 

Two variants of priority assignment:

1. Proportional to TD error: <img src="_gitex/tex_e5c41506f6c8286523d67b8647d29d31.png" alt="p_i = |\delta_i| + \epsilon"  height="16" /> where <img src="_gitex/tex_0cfc25ffaa562f52c2ac51b2d9da4b37.png" alt="\epsilon > 0"  height="12" /> ensures transitions with zero TD error will also be revisited.
2. Rank-based: <img src="_gitex/tex_f0472e41053485b0c7f427f61b55baeb.png" alt="p_i = \frac{1}{rank(i)}"  height="22" /> sorted with respect to TD error <img src="_gitex/tex_098ab9a04b9aa3f66987b3b7a9354c1d.png" alt="|\delta_i|"  height="16" />. More robust. 

Tricks must be used to efficiently compute the above two priorities (i.e. does not increase by _O(N)_).

Importance-sampling weights <img src="_gitex/tex_6056c278287cd1c46b7db79f99659cbe.png" alt="w_i"  height="10" /> must be used to correct the bias introduced by prioritized replay:

<img src="_gitex/tex_c4ed4de1c117074c3eb02e3860af75bb.png" alt=" w_i = (\frac{1}{N} \frac{1}{P(i)})^{-\beta} "  height="45" />

and linearly anneal <img src="_gitex/tex_d2df1bc88fc935017f9ba3c2c41ab83f.png" alt="\beta"  height="15" /> from <img src="_gitex/tex_ef1502bd58b2f08354ddcecced8cb5e1.png" alt="\beta_0"  height="15" /> at the beginning of training to 1. 

![PER](img/prioritized_experience_replay.png)

## Normalized Advantage Function (NAF)

Continuous Deep Q-Learning with Model-based Acceleration. [[arxiv '16]](http://arxiv.org/abs/1603.00748)

Formulate advantage function such that the maximum is trivial to find. 

<img src="_gitex/tex_557c74798ca1f2f20733395d43371372.png" alt=" Q(x, u) = V(x) + A(x, u) "  height="20" />

<img src="_gitex/tex_f41274d2916204258d89a3a7e301f4b5.png" alt=" A(x, u) = -\frac{1}{2} (u - \mu(x))^T P(x) (u - \mu(x)) "  height="40" />

All the above functions _Q_, _V_, _A_, _P_ are parameterized. 

<img src="_gitex/tex_8a19c1b51bd0f75541d7d3a804117416.png" alt="P(x)"  height="17" /> is a state-dependent, positive-definite square matrix where <img src="_gitex/tex_8bcf3b7eaf6701a4b07f7d953f811695.png" alt="P(x; \theta^P) = L(x; \theta^P) L(x; \theta^P)^T"  height="17" /> and <img src="_gitex/tex_ced227530642eba53120e69161d078a8.png" alt="L(x)"  height="17" /> is a lower-triangular matrix whose entries come from a linear output layer of a neural network, with the diagonal terms exponentiated.

The action that maximizes _Q_ is always <img src="_gitex/tex_9f881e772f191cb912a960ca78aa0aa8.png" alt="\mu(x; \theta^\mu)"  height="17" />.

![NAF](img/NAF.png)

## Bootstrapped DQN

Deep Exploration via Bootstrapped DQN. [[arxiv '16]](http://arxiv.org/abs/1602.04621)

![bootstrapped](img/Bootstrapped_DQN.png)
