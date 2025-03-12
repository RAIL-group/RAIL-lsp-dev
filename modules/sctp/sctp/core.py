from enum import Enum
from sctp import sctp_graphs, robot
import numpy as np
from dataclasses import dataclass
from sctp.param import EventOutcome, APPROX_TIME, STUCK_COST, RobotType


class Action(object):
    def __init__(self, target, rtype=RobotType.Ground, start_pose = (0.0,0.0)):
        self.target = target # vertex id
        self.rtype=rtype
        self.start_pose = start_pose
    def update_pose(self, pose):
        self.start_pose = pose
    def __eq__(self, other):
        return self.target == other.target
    def __hash__(self):
        return hash(self.target)
    def __str__(self):
        if self.rtype == RobotType.Ground:
            return f'Robot goes from {self.start_pose[0], self.start_pose[1]} to V{self.target}'
        return f'Drone goes from {self.start_pose[0], self.start_pose[1]} to V{self.target}'

class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_history(self, action, outcome):
        assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
        self._data[action] = outcome


    def get_action_outcome(self, action):
        # return the history or, it it doesn't exist, return CHANCE
        return self._data.get(action, EventOutcome.CHANCE)

    def copy(self):
        return History(data=self._data.copy())
    
    def get_data_length(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return self._data == other._data

    def __str__(self):
        return f'{self._data}'

    def __hash__(self):
        return hash(tuple(self._data.items()))

# def get_state_from_history(outcome_states, history):
#     for state in outcome_states:
#         if state.history == history:
#             return state

def get_edge(edges, v1_id, v2_id):
    for edge in edges:
        if (edge.v1.id == v1_id and edge.v2.id == v2_id) or (edge.v1.id == v2_id and edge.v2.id == v1_id):
            return edge
    raise ValueError(f'Edge between {v1_id} and {v2_id} not found')
    

@dataclass
class sctp_metric:
    elap_time: float
    trav_dist: float

    def __lt__(self, other):
        return (self.elap_time, self.trav_dist) < (other.elap_time, other.trav_dist)
    def __eq__(self, other):
        return (self.elap_time, self.trav_dist) == (other.elap_time, other.trav_dist)
    def __gt__(self, other):
        return (self.elap_time, self.trav_dist) > (other.elap_time, other.trav_dist)
    def __add__(self, other):
        return sctp_metric(self.elap_time + other.elap_time, self.trav_dist + other.trav_dist)
class SCTPState(object):
    def __init__(self, graph=None, last_state=None, goal=None, robots=None):
        self.action_cost = 0.0
        self.heuristic = 0.0
        self.noway2goal = False      
        if last_state is None:
            self.history = History()
            for vertex in graph.vertices:
                action = Action(target=vertex.id)
                self.history.add_history(action, EventOutcome.TRAV)
            self.poi_probs = {poi: poi.block_prob for poi in graph.pois}
            self.edge_costs = {edge: edge.cost for edge in graph.edges}
            self.vertices = graph.vertices
            self.edges = graph.edges
            self.pois = graph.pois
            self.goalID = goal
            self.robot = robots[0]
            self.robot.need_action = True
            self.uavs = robots[1:]
            for uav in self.uavs:
                uav.need_action = True
            self.uav_actions = [Action(target=poi.id, rtype=RobotType.Drone) for poi in self.pois] # list unexplored pois
            assert self.robot.edge is None
            neighbors = [node for node in self.vertices+self.pois if node.id == self.robot.last_node][0].neighbors
            self.robot_actions = [Action(target=neighbor) for neighbor in neighbors]
            self.state_actions = [action for action in self.uav_actions]    
            self.assigned_pois = set()
            self.gateway = set()
            for v in self.vertices+self.pois:
                if v.id == self.robot.last_node:
                    [self.gateway.add(nei) for nei in v.neighbors]
            self.v_vertices = set()
            self.v_vertices.add(self.robot.last_node)
        else:
            self.history = last_state.history.copy()
            self.poi_probs = last_state.poi_probs
            self.edge_costs = last_state.edge_costs
            self.vertices = last_state.vertices
            self.edges = last_state.edges
            self.pois = last_state.pois
            self.goalID = last_state.goalID
            self.robot = last_state.robot.copy()
            self.uavs = [uav.copy() for uav in last_state.uavs]
            self.uav_actions = [Action(target=action.target, rtype=action.rtype) for action in last_state.uav_actions]
            self.robot_actions = [Action(target=action.target, rtype=action.rtype) for action in last_state.robot_actions]
            # self.action_cost = 0.0
            self.assigned_pois = last_state.assigned_pois.copy() # [poi for poi in last_state.assigned_pois]
            self.state_actions = []
            self.gateway = last_state.gateway.copy()
            self.v_vertices = last_state.v_vertices.copy()
            self.noway2goal = last_state.noway2goal
        # self.cal_heuristic() 

    def get_actions(self):
        return self.state_actions

    # need to work on this - the heuristic is changed
    # def cal_heuristic(self): 
    #     if self.robot.at_node:
    #         self.heuristic = [node for node in self.vertices + self.pois \
    #                           if node.id == self.robot.last_node][0].heur2goal
    #     else:
    #         node1_id = self.robot.edge[0]
    #         node2_id = self.robot.edge[1]
    #          = [node for node in self.vertices + self.pois \
    #                           if node.id == self.robot.last_node][0].heur2goal
    @property
    def is_goal_state(self):
        return self.robot.last_node == self.goalID or self.noway2goal


    def transition(self, action):
        temp_state = self.copy()
        uav_needs_action = [uav.need_action for uav in temp_state.uavs]
        anyrobot_action = False
        # print("+++++++++++++++++++++++++++++++++++++++++++++++")
        
        if action.rtype == RobotType.Drone:
            # print(f"Assigning action: {action} to drone")
            assert any(uav_needs_action) == True
            assert action in temp_state.uav_actions
            uav_idx = uav_needs_action.index(True)
            start_pos = temp_state.uavs[uav_idx].cur_pose
            action.update_pose(start_pos)
            end_pos = [node for node in temp_state.vertices+temp_state.pois if node.id == action.target][0].coord
            if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
                ValueError("Start position is NaN") 
            distance = np.linalg.norm(start_pos - np.array([end_pos[0], end_pos[1]]))
            if distance != 0.0:
                direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
            else:
                direction = np.array([1.0, 1.0])
            temp_state.uavs[uav_idx].retarget(action, distance, direction)
            if action.target not in temp_state.assigned_pois:
                temp_state.assigned_pois.add(action.target)
            anyrobot_action = True
        elif action.rtype == RobotType.Ground:
            # print(f"Assigning action: {action} to robot")
            assert temp_state.robot.need_action == True
            start_pos = temp_state.robot.cur_pose
            action.update_pose(start_pos)
            end_pos = [node for node in temp_state.vertices+temp_state.pois if node.id == action.target][0].coord
            distance = np.linalg.norm(start_pos - np.array(end_pos))
            if distance != 0.0:
                direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
            else:
                direction = np.array([1.0, 1.0])
            temp_state.robot.retarget(action, distance, direction)
            anyrobot_action = True
        assert anyrobot_action == True
        # print("##########################################")
        return advance_state(temp_state, action)

    def copy(self):
        return SCTPState(last_state=self)

def advance_state(state1, action):
    state = state1.copy()
    # 1. if any robot needs action, determine its actions then return
    if any([uav.need_action for uav in state.uavs]):
        state.state_actions = [action for action in state.uav_actions]
        return {state: (1.0, 0.0)}
    if state.robot.need_action:
        # set action for this state
        state.state_actions = [Action(target=action.target, rtype=action.rtype) for action in state.robot_actions]
        stuck = is_robot_stuck(state)
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, uav_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_node = state.robot.last_node
    edge = state.robot.edge
    # move the robots
    state.robot.advance_time(time_advance)
    for uav in state.uavs:
        if uav.last_node != state.goalID:
            uav.advance_time(time_advance)
        else:
            # print("Drone is at goal - no move")
            uav.need_action = False
    if robot_reach_first:
        # print(f"Robot finishes its action {state.robot.action} with time {time_advance}")
        vertex_status = state.history.get_action_outcome(state.robot.action)
        vertex = [node for node in state.vertices+state.pois if node.id == state.robot.action.target][0]
        # update the uav action set.
        state.uav_actions = [action for action in state.uav_actions if action.target != state.robot.last_node]
        visiting = False
        state.gateway.discard(state.robot.last_node)
        if state.robot.last_node not in state.v_vertices:
            state.v_vertices.add(state.robot.last_node)
            visiting = True            
        if vertex_status == EventOutcome.BLOCK:
            # update the ground robot action set - only available action is going back.
            if edge is None: # the robot starts its action from a node
                state.robot_actions = [Action(target=last_node)]
            else: # the robot is on an edge as starting its action
                state.robot_actions = [Action(target = edge[0])]
            state.state_actions = [action for action in state.robot_actions]
            return {state: (1.0, state.action_cost)}
        # if edge_status is traversable, return action traversable (state) with traversable cost
        elif vertex_status == EventOutcome.TRAV: 
            neighbors = [node for node in state.vertices+state.pois if node.id == state.robot.action.target][0].neighbors
            state.robot_actions = [Action(target=neighbor) for neighbor in neighbors]
            state.robot_actions = [action for action in state.robot_actions \
                                    if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
            assert state.robot.last_node == state.robot.action.target
            if visiting:
                add_gateway(state)
            state.state_actions = [action for action in state.robot_actions ]
            stuck = is_robot_stuck(state)
            return {state: (1.0, state.action_cost)}
        # if edge_status is 'CHANCE', we don't know the outcome.action
        elif vertex_status == EventOutcome.CHANCE:        
            reset_uavs_action(state)
            # TRAVERSABLE
            new_state_trav = state.copy()
            new_state_trav.action_cost = state.action_cost
            new_state_trav.history.add_history(state.robot.action, EventOutcome.TRAV)
            neighbors = [node for node in state.vertices+state.pois if node.id == state.robot.action.target][0].neighbors
            new_state_trav.robot_actions = [Action(target=neighbor) for neighbor in neighbors if last_node !=neighbor]
            new_state_trav.robot_actions = [action for action in new_state_trav.robot_actions \
                                                if new_state_trav.history.get_action_outcome(action) != EventOutcome.BLOCK]
            if visiting:
                add_gateway(new_state_trav)
            stuck = is_robot_stuck(new_state_trav)
            new_state_trav.state_actions = [action for action in new_state_trav.robot_actions]
            # BLOCKED
            new_state_block = state.copy()
            new_state_block.action_cost = state.action_cost
            new_state_block.history.add_history(state.robot.action, EventOutcome.BLOCK)
            if edge is None:
                new_state_block.robot_actions = [Action(target=last_node)]
            else:
                new_state_block.robot_actions = [Action(target=edge[0])]
            new_state_block.state_actions = [action for action in new_state_block.robot_actions]
            stuck= is_robot_stuck(new_state_block)
                # print("robot gets stuck -- terminal state")

            return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                        new_state_block: (vertex.block_prob, new_state_block.action_cost)}
    else:
        # print(f"Drone finishes its action {state.uavs[0].action} with time {time_advance}")
        vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
        vertex = [node for node in state.pois+state.vertices if node.id == state.uavs[uav_index].action.target][0]
        # determine all actions related to the poi and remove it from the set.
        poi_id = state.uavs[uav_index].last_node
        assert poi_id == vertex.id
        state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
        if len(state.uav_actions) == 0:
            state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
            state.state_actions = [action for action in state.uav_actions]
        else:
            state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
        
        # if some uavs also finish theirs actions, reset need_action for the next iteration
        for i, uav in enumerate(state.uavs):
            if i != uav_index and uav.need_action: # and uav.last_node != state.goalID:
                uav.need_action = False 
        if state.robot.at_node:
            state.robot.need_action = False 
        
        if vertex_status == EventOutcome.BLOCK: # should not go here
            ValueError("Drones should never visit this node")
            return {state: (1.0, state.action_cost)}
        elif vertex_status == EventOutcome.TRAV: 
            assert vertex.id == state.goalID
            return {state: (1.0, state.action_cost)}
        elif vertex_status == EventOutcome.CHANCE:
            if not state.robot.at_node: # reset if it is in middle of action
                state.robot.need_action = True 
                state.robot.remaining_time = 0.0
                state.robot_actions = [Action(target=state.robot.edge[0]), Action(target=state.robot.edge[1])]
            
            # TRAVERSABLE
            new_state_trav = state.copy()
            new_state_trav.action_cost = state.action_cost
            new_state_trav.history.add_history(state.uavs[uav_index].action, EventOutcome.TRAV)
            new_state_trav.robot_actions = [action for action in new_state_trav.robot_actions \
                                if new_state_trav.history.get_action_outcome(action) != EventOutcome.BLOCK]            
            stuck = is_robot_stuck(new_state_trav)
            new_state_trav.state_actions = [action for action in new_state_trav.uav_actions]

            # BLOCKED
            new_state_block = state.copy()
            new_state_block.action_cost = state.action_cost
            new_state_block.history.add_history(state.uavs[uav_index].action, EventOutcome.BLOCK)
            new_state_block.robot_actions = [action for action in new_state_block.robot_actions \
                                if new_state_block.history.get_action_outcome(action) != EventOutcome.BLOCK]
            new_state_block.gateway.clear()
            for gate in state.gateway:
                if new_state_block.history.get_action_outcome(Action(target=gate)) != EventOutcome.BLOCK:
                    new_state_block.gateway.add(gate)

            stuck = is_robot_stuck(new_state_block)
                # print("robot gets stuck -- terminal state")
            new_state_block.state_actions = [action for action in new_state_block.uav_actions]
            return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                        new_state_block: (vertex.block_prob, new_state_block.action_cost)}


def is_robot_stuck(state):
    if len(state.robot_actions) == 0 or len(state.gateway)==0:
        state.noway2goal = True 
        state.action_cost += STUCK_COST
        return True
    return False

def add_gateway(state):
    neighbors = [node for node in state.vertices+state.pois \
                        if node.id == state.robot.action.target][0].neighbors
    for nei in neighbors:
        act = Action(target=nei)
        if (state.history.get_action_outcome(act) != EventOutcome.BLOCK) and (nei not in state.v_vertices):
            state.gateway.add(nei)

def reset_uavs_action(state):
    for i, uav in enumerate(state.uavs):
        if not uav.need_action and uav.action not in state.uav_actions and uav.last_node != state.goalID:
            uav.need_action = True 
            uav.remaining_time = 0.0


def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    for uav in state.uavs:
        if uav.remaining_time >APPROX_TIME:
            time_remaining_uavs.append(uav.remaining_time)
    robot_reach_first = False
    if len(time_remaining_uavs)==0 or state.robot.remaining_time < min(time_remaining_uavs):
        robot_reach_first = True
        return robot_reach_first, len(time_remaining_uavs), state.robot.remaining_time
    else:
        min_time = min(time_remaining_uavs)
        remaining_times = [uav.remaining_time for uav in state.uavs]
        uav_index = remaining_times.index(min_time)
        return robot_reach_first, uav_index, min_time


def sctp_rollout(state):
    cost = 0.0
    while not state.is_goal_state:
        actions = state.get_actions()
        robot_assign = True 
        if any([uav.need_action for uav in state.uavs]):
            robot_assign = False
        action_cost = [(action, state.transition(action)) for action in actions]


        best_action = min(action_cost, key=lambda x: list(x[1].keys())[0].heuristic)
        node_action_transition = {}
        node_action_transition_cost = {}
        for state, (prob, cost) in best_action[1].items():
            node_action_transition[state] = prob
            node_action_transition_cost[state] = cost
        prob = [p for p in node_action_transition.values()]
        state = np.random.choice(list(node_action_transition.keys()), p=prob)
        cost += node_action_transition_cost[state]
    return cost

# def get_uavs_needAction(state):
#     need_actions = []
#     for i, uav in enumerate(state.uavs):
#         if uav.need_action and uav.last_node != state.goalID:
#             need_actions.append(i)
#     return need_actions

# def get_edge_from_action(state, action):
#    v1 = [v for v in state.vertices if v.id == action.start][0]
#    v2 = [v for v in state.vertices if v.id == action.end][0]
#    edge = [e for e in state.edges if (e.hash_id == hash(v1) + hash(v2))][0]
#    return edge



# def get_action_traversability_from_history(history, action):
#    return history.get_action_outcome(action)

