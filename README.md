# Reinforcement-Learning

## Implementing DDQN for a Car Racing Game with PyTorch

This snippet demonstrates the use of Double DQN (DDQN) reinforcement learning for a car racing game developed on a learning track. The goal is to evaluate the algorithm's sample efficiency by examining the average reward achieved after a given number of episodes interacting with the environment.

For further reading, refer to the paper: [Hasselt, 2010](https://papers.nips.cc/paper/3964-double-q-learning)

## Double DQN Overview

The standard Deep Q-Network (DQN) method tends to overestimate Q-values because it uses an argmax operation over potentially biased Q-values to compute targets. This can lead to a higher likelihood of selecting overestimated values.

**Standard DQN Target:**  
\[ Q(s_t, a_t) = r_t + Q(s_{t+1}, \text{argmax}_a Q(s_t, a)) \]

To mitigate this issue, Double DQN employs two uncorrelated Q-Networks. Only one Q-Network is updated through gradients, while the target Q-Network's parameters are periodically synchronized with the updated Q-Network.

**Double DQN Target:**  
\[ Q(s_t, a_t) = r_t + Q_{\theta}(s_{t+1}, \text{argmax}_a Q_{\text{target}}(s_t, a)) \]

**Loss Function:**  
\[ (Q(s_t, a_t) - Q_{\theta}(s_t, a_t))^2 \]
