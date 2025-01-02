import numpy as np
import copy
import pdb

class POUCTNode(object):
    def __init__(self, state, parent=None, action=None, cost=None):
        self.state = state
        self.parent = parent
        if self.parent is not None:
            self.cost = self.parent.cost + cost
        else:
            self.cost = 0
        self.prev_action = action
        self.children = set()
        self.unexplored_actions = [copy.copy(a) for a in self.state.get_actions()]

        # the number of times each action has been taken
        self.action_n = {act: 0 for act in self.unexplored_actions}
        # the cumulative values for each action
        self.action_values = {act: 0 for act in self.unexplored_actions}

        # to save computation, action outcome stores next node for each action from the state
        self.action_outcomes = {}

    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0

    def is_terminal_node(self):
        '''A state can have multiple actions, but can be a terminal node'''
        return len(self.state.get_actions()) == 0

    def get_best_uct_action(self, C=1.0):
        action = list(self.action_n.keys())
        action_values = np.array([self.action_values[a] for a in action])
        action_n = np.array([self.action_n[a] for a in action])
        uct_values = (-1) * action_values/action_n + C * np.sqrt(np.log(np.sum(action_n))/action_n)
        return action[np.argmax(uct_values)]

def po_mcts(state, n_iterations=1000, C=10.0, rollout_fn=None):
    root = POUCTNode(state)
    for _ in range(n_iterations):
        leaf = traverse(root, C=C)
        simulation_result = rollout(leaf, rollout_fn=rollout_fn)
        backpropagate(leaf, simulation_result)
    best_action, cost = get_best_action(root)
    path = get_best_path(leaf)
    print(path)
    return best_action, cost

def traverse(node, C=1.0):
    while node.is_fully_expanded() and not node.is_terminal_node():
        action = node.get_best_uct_action(C=C)
        child_node = get_chance_node(node, action)
        if child_node not in node.children:
            return child_node
        else:
            node = child_node

    if node.is_terminal_node():
        return node

    # 1. pick a new action
    action = node.unexplored_actions.pop()
    # 2. create a new node
    new_child = get_chance_node(node, action)
    # 3. add to the children
    node.children.add(new_child)
    return new_child


def rollout(node, rollout_fn=None):
    if rollout_fn is not None:
        return node.cost + rollout_fn(node.state)
    else:
        # do a random rollout
        rollout_cost = 0
        while not node.is_terminal_node():
            action = np.random.choice(node.unexplored_actions)
            node = get_chance_node(node, action)
        return node.cost + rollout_cost

def backpropagate(node, result):
    if node.parent is not None:
        node.parent.action_n[node.prev_action] += 1
        node.parent.action_values[node.prev_action] += result
        backpropagate(node.parent, result)

def get_best_action(node):
    actions = list(node.action_n.keys())
    action_values = [node.action_values[a] for a in actions]
    action_n = [node.action_n[a] for a in actions]

    # get all index with highest action_n
    max_n = np.max(action_n)
    best_action_idxs = [i for i, n in enumerate(action_n) if n == max_n]

    # if there are multiple actions with the same number of visits, choose the one with the lowest cost
    if len(best_action_idxs) > 1:
        best_action_idx = min(best_action_idxs, key=lambda x: action_values[x])
    else:
        best_action_idx = best_action_idxs[0]

    best_action = actions[best_action_idx]
    best_action_cost = action_values[best_action_idx] / action_n[best_action_idx]
    return best_action, best_action_cost

def get_chance_node(node, action):
    if action in node.action_outcomes:
        node_action_transition = node.action_outcomes[action]
    else:
        state_prob_cost = node.state.transition(action)
        node_action_transition = {}
        for state, (prob, cost) in state_prob_cost.items():
            child_node = POUCTNode(state, parent=node, action=action, cost=cost)
            node_action_transition[child_node] = prob
        node.action_outcomes[action] = node_action_transition

    prob = [p for p in node_action_transition.values()]
    chance_node = np.random.choice(list(node_action_transition.keys()), p=prob)
    return chance_node

def get_best_path(node):
    path = []
    while node.parent is not None:
        path.append(node.prev_action)
        node = node.parent
    return path[::-1]

def get_best_path_from_root(root):
    path = []
