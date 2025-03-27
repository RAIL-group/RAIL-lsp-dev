from enum import Enum
from sctp import sctp_graphs, robot
from pouct_planner import core as pomcp
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
    def __init__(self, graph=None, goalID=None, robot=None, drones=None, iscopy=False):
        self.action_cost = 0.0
        self.heuristic = 0.0
        self.noway2goal = False
        self.depth = 0
              
        if not iscopy:
            self.max_depth = 10*(len(graph.pois) + len(graph.vertices))
            self.history = History()
            for vertex in graph.vertices+graph.pois:
                action = Action(target=vertex.id)
                if vertex.block_prob == 1.0:
                    self.history.add_history(action, EventOutcome.BLOCK)
                elif vertex.block_prob == 0.0:
                    self.history.add_history(action, EventOutcome.TRAV)
            self.poi_probs = {poi: poi.block_prob for poi in graph.pois}
            self.edge_costs = {edge: edge.cost for edge in graph.edges}
            self.vertices = graph.vertices
            self.edges = graph.edges
            self.pois = graph.pois
            self.goalID = goalID
            self.robot = robot
            self.robot.need_action = True
            self.uavs = drones
            for uav in self.uavs:
                # uav.need_action = False if uav.last_node == self.goalID else True
                # # if uav.last_node == self.goalID:
                # #     uav.need_action = False 
                # # else:
                uav.need_action = True
            self.assigned_pois = set()
            self.gateway = set()
            self.v_vertices = set()
            self.heuristic_vertices = dict()
            self.uav_actions = None 
            if drones:
                self.uav_actions = [Action(target=poi.id, rtype=RobotType.Drone) for poi in self.pois]
                self.uav_actions = [action for action in self.uav_actions \
                                    if self.history.get_action_outcome(action) == EventOutcome.CHANCE] # list unexplored pois
                if len(self.uav_actions) == 0:
                    self.uav_actions =[Action(target=self.goalID,rtype=RobotType.Drone)]
            if self.robot.at_node:
                neighbors = [node for node in self.vertices+self.pois if node.id == self.robot.last_node][0].neighbors
                self.robot_actions = [Action(target=neighbor, start_pose=self.robot.cur_pose) for neighbor in neighbors]
                self.v_vertices.add(self.robot.last_node)
                # print("The initial robot actions:")
                # print([action.target for action in self.robot_actions])
                # print(f"The last visited vertex: {self.robot.v_vertices}")
                if self.history.get_action_outcome(Action(target=self.robot.last_node))==EventOutcome.BLOCK:
                    self.robot_actions = [action for action in self.robot_actions if action.target ==self.robot.v_vertices]
                [self.gateway.add(action.target) for action in self.robot_actions]
                
                
                self.heuristic_vertices[self.robot.last_node] = [v.heur2goal for v in self.vertices+self.pois \
                                                              if v.id == self.robot.last_node][0]
                self.heuristic = self.heuristic_vertices[self.robot.last_node]
            else:
                self.robot_actions = [Action(target=self.robot.edge[0], start_pose=self.robot.cur_pose), 
                                      Action(target=self.robot.edge[1],start_pose=self.robot.cur_pose)]
                [self.gateway.add(nodeid) for nodeid in self.robot.edge \
                    if self.history.get_action_outcome(Action(target=nodeid)) != EventOutcome.BLOCK]
                n1 = [v for v in self.vertices+self.pois if v.id == self.robot.edge[0]][0]
                n2 = [v for v in self.vertices+self.pois if v.id == self.robot.edge[1]][0]
                self.heuristic = min(n1.heur2goal+np.linalg.norm(self.robot.cur_pose - np.array(n1.coord)),
                                     n2.heur2goal+np.linalg.norm(self.robot.cur_pose - np.array(n2.coord)))
            # print("The robot actions before filtering:")
            # print([action.target for action in self.robot_actions])
            
            self.robot_actions = [action for action in self.robot_actions \
                                  if self.history.get_action_outcome(action) != EventOutcome.BLOCK]
            assert len(self.robot_actions) > 0
            assert len(self.uav_actions) > 0
            self.state_actions = [action for action in self.uav_actions]
            # print("The robot actions after filtering:")
            # print([action.target for action in self.robot_actions])
            
    def get_actions(self):
        return self.state_actions

    # need to work on this - the heuristic is changed
    def update_heuristic(self): 
        if self.robot.at_node:
            if self.robot.last_node in self.heuristic_vertices: # is None:
                self.heuristic = self.heuristic_vertices[self.robot.last_node]+30.0
            else:
                self.heuristic = [node for node in self.vertices + self.pois \
                              if node.id == self.robot.last_node][0].heur2goal                
            self.heuristic_vertices[self.robot.last_node] = self.heuristic
            
        else:
            edge_heuristics = []
            for node_id in self.robot.edge:
                if node_id in self.heuristic_vertices:
                    edge_heu = self.heuristic_vertices[node_id]
                else:
                    edge_heu = [node for node in self.vertices + self.pois \
                              if node.id == node_id][0].heur2goal
                edge_heuristics.append(edge_heu)
            
            self.heuristic = min([edge_heuristics[i] + np.linalg.norm(self.robot.cur_pose - np.array(node.coord)) 
                                  for i, edge_id in enumerate(self.robot.edge) \
                                  for node in self.vertices + self.pois \
                                    if node.id == edge_id])
    @property
    def is_goal_state(self):
        return self.robot.last_node == self.goalID or self.noway2goal

    def transition(self, action):
        temp_state = self.copy()
        uav_needs_action = [uav.need_action for uav in temp_state.uavs]
        anyrobot_action = False
        
        if action.rtype == RobotType.Drone:
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
        return advance_state(temp_state, action)

    def copy(self):
        new_state = SCTPState(iscopy=True)
        new_state.max_depth = self.max_depth
        new_state.depth = self.depth
        new_state.history = self.history.copy()
        new_state.poi_probs = self.poi_probs
        new_state.edge_costs = self.edge_costs
        new_state.vertices = self.vertices
        new_state.edges = self.edges
        new_state.pois = self.pois
        new_state.goalID = self.goalID
        new_state.robot = self.robot.copy()
        new_state.uavs = [uav.copy() for uav in self.uavs]
        new_state.uav_actions = [Action(target=action.target, rtype=action.rtype, start_pose=action.start_pose) \
                                 for action in self.uav_actions]
        new_state.robot_actions = [Action(target=action.target, rtype=action.rtype, start_pose=action.start_pose) \
                                   for action in self.robot_actions]
        new_state.action_cost = 0.0
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.state_actions = []
        new_state.gateway = self.gateway.copy()
        new_state.v_vertices = self.v_vertices.copy()
        new_state.noway2goal = self.noway2goal
        new_state.heuristic = self.heuristic
        new_state.heuristic_vertices = self.heuristic_vertices.copy()

        return new_state

def advance_state(state1, action):
    state = state1.copy()
    # 1. if any robot needs action, determine its actions then return
    if any([uav.need_action for uav in state.uavs]):
        state.state_actions = [action for action in state.uav_actions]
        state.depth += 1
        return {state: (1.0, 0.0)}
    if state.robot.need_action:
        # set action for this state
        state.robot_actions = [action for action in state.robot_actions \
                            if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
        state.state_actions = [action for action in state.robot_actions]
        stuck = is_robot_stuck(state)
        state.depth += 1
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, uav_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_node = state.robot.last_node
    edge = state.robot.edge.copy()
    # move the robots
    state.robot.advance_time(time_advance)
    for uav in state.uavs:
        if uav.last_node != state.goalID:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
    state.update_heuristic()
    if robot_reach_first:
        return get_new_nodes_grobot(state, last_node=last_node, last_edge=edge)
    else:
        return get_new_nodes_drone(state=state, uav_index=uav_index)
        
def get_new_nodes_grobot(state, last_node, last_edge):
    vertex_status = state.history.get_action_outcome(state.robot.action)
    vertex = [node for node in state.vertices+state.pois if node.id == state.robot.action.target][0]
    # update the uav action set.
    state.uav_actions = [action for action in state.uav_actions if action.target != state.robot.last_node]
    if len(state.uav_actions) == 0:
        state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
    visiting = False
    state.gateway.discard(state.robot.last_node)
    if state.robot.last_node not in state.v_vertices:
        state.v_vertices.add(state.robot.last_node)
        visiting = True            
    if vertex_status == EventOutcome.BLOCK:
        # update the ground robot action set - only available action is going back.
        if state.robot.at_node: #last_edge is None: # the robot starts its action from a node
            state.robot_actions = [Action(target=last_node, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        else: # the robot is on an edge as starting its action
            state.robot_actions = [Action(target = last_edge[0],start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        state.state_actions = [action for action in state.robot_actions]
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    # if edge_status is traversable, return action traversable (state) with traversable cost
    elif vertex_status == EventOutcome.TRAV: 
        neighbors = [node for node in state.vertices+state.pois if node.id == state.robot.last_node][0].neighbors
        state.robot_actions = [Action(target=neighbor, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) for neighbor in neighbors]
        state.robot_actions = [action for action in state.robot_actions \
                                if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
        assert state.robot.last_node == state.robot.action.target
        if visiting:
            add_gateway(state)
        state.state_actions = [action for action in state.robot_actions ]
        stuck = is_robot_stuck(state)
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    # if edge_status is 'CHANCE', we don't know the outcome.action
    elif vertex_status == EventOutcome.CHANCE:
        reset_uavs_action(state)
        # TRAVERSABLE
        new_state_trav = state.copy()
        new_state_trav.action_cost = state.action_cost
        new_state_trav.history.add_history(state.robot.action, EventOutcome.TRAV)
        neighbors = [node for node in state.vertices+state.pois if node.id == state.robot.last_node][0].neighbors
        new_state_trav.robot_actions = [Action(target=neighbor,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) for neighbor in neighbors if last_node !=neighbor]
        new_state_trav.robot_actions = [action for action in new_state_trav.robot_actions \
                                            if new_state_trav.history.get_action_outcome(action) != EventOutcome.BLOCK]
            
        if visiting:
            add_gateway(new_state_trav)
        stuck = is_robot_stuck(new_state_trav)
        new_state_trav.state_actions = [action for action in new_state_trav.robot_actions]
        new_state_trav.depth += 1
        # BLOCKED
        new_state_block = state.copy()
        new_state_block.action_cost = state.action_cost
        new_state_block.history.add_history(state.robot.action, EventOutcome.BLOCK)
        # if new_state_block.robot.at_node:
        if last_node != new_state_block.robot.last_node:
            target = last_node
        else:
            target = new_state_block.robot.ll_node
        new_state_block.robot_actions = [Action(target=target,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        new_state_block.state_actions = [action for action in new_state_block.robot_actions]
        stuck= is_robot_stuck(new_state_block)
        new_state_block.depth += 1
        assert new_state_block.depth == new_state_trav.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def get_new_nodes_drone(state, uav_index):
    vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
    vertex = [node for node in state.pois+state.vertices if node.id == state.uavs[uav_index].action.target][0]
    # determine all actions related to the poi and remove it from the set.
    poi_id = state.uavs[uav_index].last_node
    assert poi_id == vertex.id
    state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
    if len(state.uav_actions) == 0:
        state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone, 
                                    start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
        state.state_actions = [action for action in state.uav_actions]
    else:
        for action in state.uav_actions:
            action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
        state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
    
    # if some uavs also finish theirs actions, reset need_action for the next iteration
    for i, uav in enumerate(state.uavs):
        if i != uav_index and uav.need_action:
            uav.need_action = False 
    if state.robot.at_node:
        state.robot.need_action = False 
    
    if vertex_status == EventOutcome.BLOCK: # should not go here
        ValueError("Drones should never visit this node")
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == EventOutcome.TRAV: 
        assert vertex.id == state.goalID
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == EventOutcome.CHANCE:
        if not state.robot.at_node: # reset if it is in middle of action
            state.robot.need_action = True 
            state.robot.remaining_time = 0.0
            state.robot_actions = [Action(target=state.robot.edge[0], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1])), 
                                   Action(target=state.robot.edge[1], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1]))]
            state.robot_actions = [action for action in state.robot_actions \
                                        if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
        
        # TRAVERSABLE
        new_state_trav = state.copy()
        new_state_trav.action_cost = state.action_cost
        new_state_trav.history.add_history(state.uavs[uav_index].action, EventOutcome.TRAV)
        stuck = is_robot_stuck(new_state_trav)
        new_state_trav.depth += 1
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
        new_state_block.state_actions = [action for action in new_state_block.uav_actions]
        new_state_block.depth += 1
        assert new_state_trav.depth == new_state_block.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}


def is_robot_stuck(state):
    if state.robot.last_node == state.goalID:
        return False
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
        assert len(actions) > 0
        rtype = actions[0].rtype
        # compute the best action depending on drone or ground robot
        if rtype == RobotType.Ground:
            best_action = np.random.choice(actions)
            best_states = state.transition(best_action)
            # state_prob_costs = [(action, state.transition(action)) for action in actions]
            # best_states = min(state_prob_costs, key=lambda x: list(x[1].keys())[0].heuristic + list(x[1].keys())[0].action_cost)
            # best_states = best_states[1]
        elif rtype==RobotType.Drone: # random selection
            best_action = np.random.choice(actions)
            best_states = state.transition(best_action)
        else:
            TypeError("robot type is wrong")

        node_action_transition = {}
        node_action_transition_cost = {}
        for state, (prob, cost) in best_states.items():
            node_action_transition[state] = prob
            node_action_transition_cost[state] = cost
        prob = [p for p in node_action_transition.values()]
        state = np.random.choice(list(node_action_transition.keys()), p=prob)
        cost += node_action_transition_cost[state]
    return cost

def sctp_rollout2(node):    
    count = 0
    while not node.is_terminal_node():
        count += 1
        # print(f"Not a terminal node {count} with gateway {node.state.gateway}")
        
        # print(f"The status of the POI5 {node.state.history.get_action_outcome(act5)}")
        # print(f"The status of the POI8 {node.state.history.get_action_outcome(act8)}")
        # print(f"The status of the POI6 {node.state.history.get_action_outcome(act6)}")
        actions = node.unexplored_actions
        assert len(actions) > 0
        rtype = actions[0].rtype
        # compute the best actiFinding rollout"on depending on drone or ground robot
        if rtype == RobotType.Ground:
            # state_prob_costs = [(action, pomcp.get_chance_node(node, action)) for action in actions]
            # best_stComputing joint actioComputing joint actioates = min(state_prob_costs, key=lambda x: x[1].state.heuristic + x[1].state.action_cost)
            # action = best_states[0]
            action = np.random.choice(actions)
        elif rtype==RobotType.Drone: # random selection
            action = np.random.choice(actions)
        else:
            TypeError("robot type is wrong")
        # if 2 in list(node.state.gateway):
        # print(f"The action is: {action} with count = {count} and gateway {list(node.state.gateway)}")
        # if len(node.state.gateway)==1 and list(node.state.gateway)[0] == 2:
        #     act5 = Action(target=5)
        #     act8 = Action(target=8)
        #     act6 = Action(target=6)
            
        #     if node.state.history.get_action_outcome(act5) == EventOutcome.BLOCK and node.state.history.get_action_outcome(act6) == EventOutcome.BLOCK:
        #         print(f"The action is: {action}")
        #         print(f"count = {count} and the robot get stuck")
        #         exit()

        node = pomcp.get_chance_node(node, action)
    # print("Out of rln loop -------------------")
    # Decomment if doing rollout check
    # if node.state.noway2goal:
    #     print(f"The robot gets stuck ---- tree depth {count}")
    # if node.state.robot.last_node == node.state.goalID:
    #     print(f"The robot reaches the goal ----- tree depth {count}")
    return node.cost
