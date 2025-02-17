from enum import Enum
from sctp import sctp_graphs, robot
import numpy as np
from dataclasses import dataclass
from sctp.param import EventOutcome, BLOCK_COST, STUCK_COST


class Action(object):
    def __init__(self, start, end):
        self.start = start # vertex id
        self.end = end # vertex id
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    def __hash__(self):
        return hash(self.end)
    def __str__(self):
        return f'Action: ({self.start} -> {self.end})'

class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_history(self, vertex, outcome):
        assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
        self._data[vertex] = outcome


    def get_action_outcome(self, vertex):
        # return the history or, it it doesn't exist, return CHANCE
        return self._data.get(vertex, EventOutcome.CHANCE)

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
    def __init__(self, graph=None, last_state=None, history=None, 
                goal=None, robots=None):
        self.action_cost = sctp_metric(0.0, 0.0)
        # self.targeting = False
        if history is None:
            self.history = History()
            for vertex in graph.vertices:
                # for neighbor in vertex.neighbors:
                self.history.add_history(vertex.id, EventOutcome.TRAV)
        else:
            self.history = history.copy()
        if last_state is None:
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
            self.uav_actions = [poi.id for poi in self.pois] # list unexplored pois
            neighbors = [node for node in self.vertices+self.pois if node.id == self.robot.cur_node][0].neighbors
            self.robot_actions = [Action(start=self.robot.cur_node, end=neighbor)
                        for neighbor in neighbors]
        else:
            # how time elapsing is handled?
            self.poi_probs = last_state.poi_probs
            self.edge_costs = last_state.edge_costs
            self.vertices = last_state.vertices
            self.edges = last_state.edges
            self.pois = last_state.pois
            self.goalID = last_state.goalID
            self.robot = robot.Robot(position=last_state.G_robot.cur_pose, cur_node=last_state.G_robot.cur_node)
            self.uavs = [robot.Robot(position=robot.cur_pose, cur_node=robot.cur_node, robot_type=robot.robot_type) for robot in last_state.D_robots]
            self.uav_actions = [poi_id for poi_id in last_state.uav_actions]

        self.heuristic = 0.0
        self.update_heuristic()
        
    def get_robot_actions(self):
        return self.robot_actions

    def get_uav_actions(self):
        return self.uav_actions    
    def update_heuristic(self):
        for node in self.vertices + self.pois:
            if node.id == self.robot.cur_node:
                self.heuristic = node.heur2goal
                break

    @property
    def is_goal_state(self):
        return self.robot.cur_node == self.goalID


    def transition(self, action):
        temp_state = self.copy()
        uav_needs_action = [uav.need_action for uav in temp_state.uavs]
        
        start_pos = [node for node in temp_state.vertices+temp_state.pois if node.id == action.start][0].coord
        end_pos = [node for node in temp_state.vertices+temp_state.pois if node.id == action.end][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        direction = (np.array(end_pos) - np.array(start_pos))/distance
        if any(uav_needs_action):
            # temp_state.targeting = True
            temp_state.uavs[uav_needs_action.index(True)].retarget(action, distance, direction)
        elif self.robot.need_action:
            # temp_state.targeting = True
            temp_state.robot.retarget(action, distance, direction)

        return advance_state(temp_state, action)

    def copy(self):
        return SCTPState(last_state=self, history=self.history.coy())
 
    def hash_robot(self):
        return self.robot + sum([hash(uav) for uav in self.uavs])

    def hash_state(self):
        robot_hash = self.hash_robot()
        combined_hash = hash((robot_hash, self.history))
        return combined_hash

    def __hash__(self):
        self.hash_id = self.hash_state()
        return self.hash_id

    def __eq__(self, other):
        return self.hash_id == other.hash_id


def advance_state(state, action, prob=1.0, cost=0.0):
    # 1. if any robot needs action, return
    if state.robot.need_action or any([uav.need_action for uav in state.uavs]):
        # state.targeting = False
        return {state: (prob, cost)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, uav_index, time_advance = _get_robot_that_finishes_first(state)
    
    state.robot.advance_time(time_advance)
    for uav in state.uavs:
        uav.advance_time(time_advance)
    if robot_reach_first:
        vertex_status = state.history.get_action_outcome(state.robot.action)
        vertex = [node for node in state.vertices+state.pois if node.id == state.robot.action][0]
        block_prob = vertex.block_prob
        # if edge_status is blocked, return action blocked (state) with blocked cost
        if vertex_status == EventOutcome.BLOCK or block_prob == 1.0:
            state.robot.need_action = True
            if block_prob == 1.0:
                state.history.add_history(state.robot.action, EventOutcome.BLOCK)
            # new_state_block.robot_move(action)
            state.action_cost.elap_time = time_advance

            # update the travel distance of uavs
            
            # and the action set.



            return {state: (1.0, cost)}

        # if edge_status is traversable, return action traversable (state) with traversable cost
        elif edge_status == EventOutcome.TRAV or block_prob == 0.0:
            if edge_status == EventOutcome.TRAV:            
                new_state_trav = SCTPBaseState(last_state=state, history=state.history)
            else:
                trav_history = state.history.copy()
                trav_history.add_history(action, EventOutcome.TRAV)
                new_state_trav = SCTPBaseState(last_state=state, history=trav_history)
            new_state_trav.robot_move(action)
            new_state_trav.action_cost = state.edge_costs[edge]
            # if new_state_trav.get_actions() == [] and not new_state_trav.is_goal_state:
            #     new_state_trav.action_cost += STUCK_COST
            return {new_state_trav: (1.0, new_state_trav.action_cost)}

        # if edge_status is 'CHANCE', we don't know the outcome.
        elif edge_status == EventOutcome.CHANCE:        
            # TRAVERSABLE
            trav_history = state.history.copy()
            trav_history.add_history(action, EventOutcome.TRAV)
            new_state_trav = SCTPBaseState(last_state=state, history=trav_history)
            new_state_trav.robot_move(action)
            new_state_trav.action_cost = state.edge_costs[edge]
            # if new_state_trav.get_actions() == [] and not new_state_trav.is_goal_state:
            #     new_state_trav.action_cost += STUCK_COST
            # BLOCKED
            block_history = state.history.copy()
            block_history.add_history(action, EventOutcome.BLOCK)
            new_state_block = SCTPBaseState(last_state=state, history=block_history)
            new_state_block.robot_move(action)
            new_state_block.action_cost = BLOCK_COST

            return {new_state_trav: (1.0-block_prob, new_state_trav.action_cost),
                        new_state_block: (block_prob, BLOCK_COST)}
    else:
        pass
#     edge = get_edge_from_action(state, action)


def get_edge_from_action(state, action):
   v1 = [v for v in state.vertices if v.id == action.start][0]
   v2 = [v for v in state.vertices if v.id == action.end][0]
   edge = [e for e in state.edges if (e.hash_id == hash(v1) + hash(v2))][0]
   return edge

# def get_next_event_and_time(robot, history):
#     return robot.time_remaining
    
def _get_robot_that_finishes_first(state):
    time_remaining_uav = [uav.time_remaining for uav in state.uavs]
    robot_reach_first = False
    if state.robot.time_remaining < min(time_remaining_uav):
        robot_reach_first = True
        return robot_reach_first, len(time_remaining_uav), state.robot.time_remaining
    else:
        uav_index = time_remaining_uav.index(min(time_remaining_uav))
        return robot_reach_first, uav_index, time_remaining_uav[uav_index]



# def get_action_traversability_from_history(history, action):
#    return history.get_action_outcome(action)

# def sctpbase_rollout(state):
#     cost = 0.0
#     while not (state.is_goal_state or len(state.get_actions()) == 0):
#         actions = state.get_actions()
#         action_cost = [(action, state.transition(action)) for action in actions]
#         best_action = min(action_cost, key=lambda x: list(x[1].keys())[0].heuristic)
#         node_action_transition = {}
#         node_action_transition_cost = {}
#         for state, (prob, cost) in best_action[1].items():
#             node_action_transition[state] = prob
#             node_action_transition_cost[state] = cost
#         prob = [p for p in node_action_transition.values()]
#         state = np.random.choice(list(node_action_transition.keys()), p=prob)
#         cost += node_action_transition_cost[state]
#     return cost
