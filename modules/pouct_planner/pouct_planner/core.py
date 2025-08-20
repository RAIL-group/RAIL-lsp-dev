import numpy as np
import copy
import pdb
from sctp.param import STUCK_COST, EventOutcome, RobotType
from pathlib import Path

class POUCTNode(object):
    def __init__(self, state, parent=None, action=None, cost=None):
        self.state = state
        self.parent = parent
        if self.parent is not None:
            self.cost = self.parent.cost + cost
        else:
            self.cost = 0.0
        self.prev_action = action
        self.children = set()
        self.unexplored_actions = [copy.copy(a) for a in self.state.get_actions()]

        self.total_n = 0
        # the number of times each action has been taken
        self.action_n = {act: 0 for act in self.unexplored_actions}
        # the cumulative values for each action
        self.action_values = {act: 0.0 for act in self.unexplored_actions}

        # to save computation, action outcome stores next node for each action from the state
        self.action_outcomes = {}

    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0

    def is_terminal_node(self):
        '''A state can have multiple actions, but can be a terminal node'''
        return len(self.state.get_actions()) == 0 or self.state.is_goal_state

    def get_best_uct_action(self, C=1.0):
        action = list(self.action_n.keys())
        action_values = np.array([self.action_values[a] for a in action])
        action_n = np.array([self.action_n[a] for a in action])
        uct_values = (-1.0) * action_values/action_n + C*np.sqrt(np.log(self.total_n)/action_n)
        return action[np.argmax(uct_values)]

def po_mcts(state, n_iterations=1000, C=10.0, depth=100, rollout_fn=None):
    # save_dir='/data/sctp'
    # num_drones = 1
    # logfile = Path(save_dir) / f'debug_{num_drones}.txt'
    # with open(logfile, 'w') as f:
    #     pass    
    # total_cost = 0.0
    root = POUCTNode(state)
    assert len(root.unexplored_actions) > 0
    for i in range(n_iterations):
        leaf, sa = traverse(root, C=C, max_depth=depth)
        simulation_result, g, b, rl_cost = rollout(leaf, rollout_fn=rollout_fn)
        leaf.total_n += 1
        backpropagate(leaf, simulation_result)
        
        targets = [a.target for a in sa]
        # if targets[0] == 187:
        # if targets[0] == 5 or targets[0] == 2:
        # total_cost += simulation_result
        # with open(logfile, "a+") as f:
        #     f.write(f"R-GOAL: {int(g)} | BLOCK: {int(b)} | HEU-COST: {rl_cost:7.2f} | SIM-COST: {simulation_result:7.2f} | T-COST : {total_cost:8.2f} | ACTION: {targets} \n")
    best_action, cost = get_best_action(root)
    # path_ordering, cost_ordering = get_best_path(root)
    path_ordering, cost_ordering = get_best_path_sctp(root)
    return best_action, cost, [path_ordering, cost_ordering]

def traverse(node, C=1.0, max_depth=100):
    # first_action = True
    save_action = []
    while node.is_fully_expanded() and not node.is_terminal_node():
        if node.state.depth > max_depth:
            return node, save_action
        action = node.get_best_uct_action(C=C)
        save_action.append(action) 
        child_node = get_chance_node(node, action)
        if child_node not in node.children:
            node.children.add(child_node)
            return child_node, save_action
        else:
            node = child_node
    if node.is_terminal_node():
        return node, save_action
    # 1. pick a new action
    action = node.unexplored_actions.pop()
    save_action.append(action) 
    # 2. create a new node
    new_child = get_chance_node(node, action)
    # 3. add to the children
    node.children.add(new_child)
    return new_child, save_action

def rollout(node, rollout_fn=None):
    reach_goal = False 
    block = True 
    if rollout_fn is not None:
        rollout_value = rollout_fn(node.state)
        if rollout_value == 0.0:
            reach_goal = True
            block = False
        elif rollout_value == STUCK_COST:
            block = True 
            reach_goal = False
        else:
            reach_goal = False
            block = False
        # print(f"Rollout value: {rollout_value:4.2f} | Reach goal: {int(reach_goal)} | Block: {int(block)} | cur_robot pos {node.state.robot.last_node} | goal node: {node.state.goalID}")
        return node.cost + rollout_value, reach_goal, block, rollout_value #ollout_fn(node.state)
    else:
        # do a random rollout
        rollout_cost = 0.0
        while not node.is_terminal_node():
            action = np.random.choice(node.unexplored_actions)
            node = get_chance_node(node, action)
        return node.cost + rollout_cost, reach_goal, block, rollout_cost

def backpropagate(node, result):
    if node.parent is not None:
        node.parent.total_n += 1
        node.parent.action_n[node.prev_action] += 1
        node.parent.action_values[node.prev_action] += result
        backpropagate(node.parent, result)


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

def get_best_path(root):
    paths = []
    costs = []
    node = root
    while not node.is_terminal_node():
        if node.total_n == 1 \
            or np.max([node.action_n[a] for a in list(node.action_n.keys())])==0:
            break
        best_action, cost = get_best_action(node)
        paths.append(best_action)
        costs.append(cost)
        children = list(node.action_outcomes[best_action].keys())
        node = max(children, key=lambda x: x.total_n)
        # pdb.set_trace()
    return paths, costs

def get_best_path_sctp(root):
    paths = []
    costs = []
    node = root
    for uav in root.state.uavs:
        if uav.action is not None:
            paths.append(uav.action)
    while not node.is_terminal_node():
        if node.total_n <5 \
            or np.max([node.action_n[a] for a in list(node.action_n.keys())])==0:
            break
        best_action, cost = get_best_action(node)
        paths.append(best_action)
        costs.append(cost)
        children = list(node.action_outcomes[best_action].keys())
        if root.state.uavs == []:
            node = [child for child in children if child.state.history.get_action_outcome(best_action) == EventOutcome.TRAV][0]
        else:
            node = max(children, key=lambda x: x.total_n)        
    return paths, costs
