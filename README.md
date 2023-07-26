# Q-Learning-Algorithm-Implementation-with-OpenAI-Gym

The project involves implementing the Q-learning algorithm to find the optimal policy π* for the FrozenLake environment. The algorithm initializes the Q-table arbitrarily for all state-action pairs and iteratively updates the Q-values based on the observed rewards and transitions. The Q-learning algorithm follows the following steps:

Initialize Q(s, a) arbitrarily for all s and a.
For N episodes, repeat the following:
For M episode steps, repeat the following:
Choose a random action A.
Take action A, observe reward R and new state S'.
Update Q(S, A) using the Bellman equation: Q(S, A) = R + γ * max(Q(S', a)).
Set the current state S to S'.
If S is terminal or M steps are taken, break the loop.
To evaluate the policy's performance, the project tests the current policy after every E episodes (e.g., 0, 10, 20, ...) using the eval_policy function. The eval_policy function runs a few episodes using the learned Q-values to calculate the average reward. The performance is tracked over multiple training episodes, and the average rewards are stored for analysis.

The project includes three parts:
a) Deterministic Version: The algorithm is run ten times with a fixed gamma value of 0.9 for the deterministic version of the FrozenLake environment (is_slippery=False). The average reward performance for each run is plotted on the same graph to observe how the performance improves with more training episodes.

b) Non-deterministic Version: The same algorithm is applied to the non-deterministic version of the FrozenLake environment (is_slippery=True). The average reward performance for each run is plotted to see if there are any differences compared to the deterministic version.

c) Updated Non-deterministic Version: The non-deterministic version is re-run using an updated update rule for Q-values (α = 0.1) that incorporates the learning rate. The average reward performance is plotted to compare it with the previous non-deterministic version.

Overall, the FrozenLake Q-Learning with OpenAI Gym project provides an opportunity to understand and apply the Q-learning algorithm for solving a simple grid world environment. The graphs produced allow for a visual comparison of the learning progress and performance in both deterministic and non-deterministic scenarios.
