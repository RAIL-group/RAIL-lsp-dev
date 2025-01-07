# Summary

This module introduces a Python module implementing the Partially Observable Monte Carlo Tree Search (PO-MCTS) planner. The code computes the best action and cost by taking a Markov Decision Process by handling probabilistic transitions, action costs, and goal states.

## Usage
```python
from pouct_planner import core

stochastic_mdp = {
    'S': {'A': [('S1', 0.8, 5), ('S2', 0.2, 50)]},
    'S1': {'C': [('S3', 1.0, 3)]},
    'S2': {},
    'S3': {}
}
state = MDP('S', stochastic_mdp) # This is a testing MDP state class
best_action, cost = core.po_mcts(state, n_iterations=10000, C=1.0, rollout_fn=None)
assert best_action == 'A'
assert pytest.approx(cost, abs=1.0) == (0.8 * (5 + 3) + 0.2 * 50)
```

## Requirements for State
The `state` class has some functional requirements.
```python
class State():
    def __init__(self, ...):
        self.is_goal_state = False  # Update when the state is a goal state

    def get_actions(self):
        return [action1, action2, ...]  # List of actions from the state

    def transition(self, action):
        return {State(): (prob, cost), State(): (prob, cost), ...}  # Probabilities should be normalized

    def __eq__(self, other):
        return self.hash == other.hash  # Compare states using hash

```

## Tests
- Tests for individual functions used in the planner. For eg: the average of rollout costs gives expected value,  rollout costs fall between the minimum and the maximum cost for that state, backpropagation correctly updates the node, the functionality of traversal, and so on.
- Tests for best action and costs in different MDP environments like linear deterministic MDP, large state and action spaces, stochastic MDP, and so on.


## Additional Notes
- The `rollout_fn` can be customizable. If `rollout_fn` is not provided, a random rollout is used from the current state.
- The states are sampled according to the distributions received from the `transition` function using the `get_chance_node()` function.
