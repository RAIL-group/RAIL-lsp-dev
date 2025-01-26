from enum import Enum
from sctp import graphs

EventOutcome = Enum('EventOutcome', ['BLOCK', 'TRAV','CHANCE'])
BLOCK_COST = 1e2
STUCK_COST = 5e1

class Action(object):
    def __init__(self, start_node, target_node):
        self.start = start_node
        self.end = target_node
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    def __hash__(self):
        return hash((self.start, self.end))
    def __str__(self):
        return f'Action({self.start} -> {self.end})'
class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_history(self, action, outcome):
        assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
        self._data[action] = outcome
        # the outcome is the same for the inverse action
        invert_action = Action(start_node=action.end, target_node=action.start)
        self._data[invert_action] = outcome


    def get_action_outcome(self, action):
        # return the history or, it it doesn't exist, return CHANCE
        return self._data.get(action, EventOutcome.CHANCE)

    def copy(self):
        return History(data=self._data.copy())

    def get_visited_vertices_id(self):
        visited_vertices = set()
        for action, _ in self._data.items():
            visited_vertices.add(action.start)
            visited_vertices.add(action.end)
        return visited_vertices
    
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

def get_state_from_history(outcome_states, history):
    for state in outcome_states:
        if state.history == history:
            return state


class SCTPBaseState(object):
    def __init__(self, graph=None, last_state=None, history=None, 
                goal=None, robots=None):
        self.action_cost = 0.0
        if history is None:
            self.history = History()
        else:
            self.history = history
        if last_state is None:
            self.edge_probs = {edge: edge.block_prob for edge in graph.edges}
            self.edge_costs = {edge: edge.cost for edge in graph.edges}
            self.vertices = graph.vertices
            self.edges = graph.edges
            self.goalID = goal
            self.robots = robots
        else:
            self.edge_probs = last_state.edge_probs
            self.edge_costs = last_state.edge_costs
            self.vertices = last_state.vertices
            self.edges = last_state.edges
            self.goalID = last_state.goalID
            self.robots = graphs.RobotData(last_robot=last_state.robots)
        self.visited_vertices_id = self.history.get_visited_vertices_id()
        neighbors = [node for node in self.vertices if node.id == self.robots.cur_vertex][0].neighbors
        neigh_not_visited = [neighbor for neighbor in neighbors if neighbor not in self.visited_vertices_id]
        # self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
        #                 for neighbor in neighbors]
        self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
                                for neighbor in neigh_not_visited]
    def get_actions(self):
        return self.actions

    @property
    def is_goal_state(self):
        return self.robots.cur_vertex == self.goalID


    def transition(self, action, nav=False):
        return advance_state(self, action)
        # return self.robot_transition(action, nav)


    # def robot_transition(self, action, nav=False):
    #     belief_state = {}
    #     deadend_pen = 50.0
    #     blocking_cost = 1e2
    #     edge_id = tuple(sorted((self.robots.cur_vertex, action)))
    #     block_prob = self.edge_probs[edge_id]
    #     if block_prob == 0.0:
    #         # certain for this action/edge traversable
    #         trav_history = self.history.copy()
    #         trav_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.TRAV)
    #         new_state_traversable = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=trav_history)
    #         new_state_traversable.robot_move(action, nav)
    #         new_state_traversable.action_cost = self.edge_costs[edge_id]
    #         if new_state_traversable.get_actions() == [] and not new_state_traversable.is_goal_state:
    #             # print(f'this node {action} is a deadend')
    #             new_state_traversable.action_cost += deadend_pen
    #         belief_state[new_state_traversable] = (1.0, new_state_traversable.action_cost)
    #     elif block_prob == 1.0:
    #         # certain for this action/edge blocked
    #         block_history = self.history.copy()
    #         block_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.BLOCK)
    #         new_state_block = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=block_history)
    #         new_state_block.robot_move(action, nav)
    #         new_state_block.action_cost = blocking_cost
    #         if new_state_block.get_actions() == [] and not new_state_block.is_goal_state:
    #             # print(f'this node {action} is a deadend')
    #             new_state_block.action_cost += deadend_pen
    #         belief_state[new_state_block] = (1.0, new_state_block.action_cost)
    #     else:
    #         # for blocking
    #         block_history = self.history.copy()
    #         block_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.BLOCK)

    #         new_state_block = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=block_history)
    #         new_state_block.robot_move(action, nav)
    #         new_state_block.action_cost = BLOCK_COST
    #         # if new_state_block.get_actions() == [] and not new_state_block.is_goal_state:
    #             # print(f'this node {action} is a deadend')
    #             # new_state_block.action_cost += deadend_pen
    #         belief_state[new_state_block] = (block_prob, new_state_block.action_cost)
    #         # for traversable
    #         trav_history = self.history.copy()
    #         trav_history.add_history(action, self.robots.cur_vertex, block_prob, EventOutcome.TRAV)
    #         new_state_traversable = SCTPBaseState(self.edge_probs, self.edge_costs, last_state=self, history=trav_history)
    #         new_state_traversable.robot_move(action, nav)
    #         new_state_traversable.action_cost = self.edge_costs[edge_id]
    #         if new_state_traversable.get_actions() == [] and not new_state_traversable.is_goal_state:
    #             # print(f'this node {action} is a deadend')
    #             new_state_traversable.action_cost += deadend_pen
    #         belief_state[new_state_traversable] = (1.0 - block_prob, new_state_traversable.action_cost)
    #         assert self.robots.cur_vertex != new_state_traversable.robots.cur_vertex
    #     return belief_state

    def robot_move(self, action):
        next_vertex = action.end
        node = [node for node in self.vertices if node.id == next_vertex][0]
        
        self.robots.last_vertex = self.robots.cur_vertex
        self.robots.cur_vertex = next_vertex
        self.robots.position = [node.coord[0], node.coord[1]]
        neighbors = node.neighbors
        self.visited_vertices_id.add(self.robots.cur_vertex)
        # self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
        #                 for neighbor in neighbors]
        neigh_not_visited = [neighbor for neighbor in neighbors if neighbor not in self.visited_vertices_id]
        self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
                        for neighbor in neigh_not_visited]
        
    def hash_graph(self):
        edges_tuple = tuple(self.edge_probs)
        # Compute the hash
        return hash(edges_tuple)

    def hash_robot(self):
        return hash(self.robots.cur_vertex)

    def hash_state(self):
        graph_hash = self.hash_graph()
        robot_hash = self.hash_robot()
        # Combine the two hashes
        combined_hash = hash((graph_hash, robot_hash, self.history))
        return combined_hash

    def __hash__(self):
        self.hash_id = self.hash_state()
        return self.hash_id

    def __eq__(self, other):
        return self.hash_id == other.hash_id

   # def __repr__(self):
   #    return f'{self.edge_probs, self.robots.cur_vertex}'

def advance_state(state, action):
    edge_status = state.history.get_action_outcome(action)
    edge = get_edge_from_action(state, action)
    block_prob = state.edge_probs[edge]
    # if edge_status is blocked, return action blocked (state) with blocked cost
    if edge_status == EventOutcome.BLOCK or block_prob == 1.0:
        if edge_status == EventOutcome.BLOCK:
            new_state_block = SCTPBaseState(last_state=state, history=state.history)
        else:
            block_history = state.history.copy()
            block_history.add_history(action, EventOutcome.BLOCK)
            new_state_block = SCTPBaseState(last_state=state, history=block_history)
        new_state_block.robot_move(action)
        new_state_block.action_cost = BLOCK_COST
        return {new_state_block: (1.0, BLOCK_COST)}

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
        if new_state_trav.get_actions() == [] and not new_state_trav.is_goal_state:
            new_state_trav.action_cost += STUCK_COST
        return {new_state_trav: (1.0, new_state_trav.action_cost)}

    # if edge_status is 'CHANCE', we don't know the outcome.
    elif edge_status == EventOutcome.CHANCE:        
        # TRAVERSABLE
        trav_history = state.history.copy()
        trav_history.add_history(action, EventOutcome.TRAV)
        new_state_trav = SCTPBaseState(last_state=state, history=trav_history)
        new_state_trav.robot_move(action)
        new_state_trav.action_cost = state.edge_costs[edge]
        if new_state_trav.get_actions() == [] and not new_state_trav.is_goal_state:
            new_state_trav.action_cost += STUCK_COST
        # BLOCKED
        block_history = state.history.copy()
        block_history.add_history(action, EventOutcome.BLOCK)
        new_state_block = SCTPBaseState(last_state=state, history=block_history)
        new_state_block.robot_move(action)
        new_state_block.action_cost = BLOCK_COST

        return {new_state_trav: (1.0-block_prob, new_state_trav.action_cost),
                    new_state_block: (block_prob, BLOCK_COST)}

def get_edge_from_action(state, action):
   v1 = [v for v in state.vertices if v.id == action.start][0]
   v2 = [v for v in state.vertices if v.id == action.end][0]
   edge = [e for e in state.edges if (e.hash_id == hash(v1) + hash(v2))][0]
   return edge


def get_action_traversability_from_history(history, action):
   return history.get_action_outcome(action)

def sctpbase_rollout(state):
   # not allow the robot to move back to the parent node
   # for _ in range(n_steps):
   #    if state.is_goal_state:
   #       return state
   #    action = state.get_actions()[0]
   #    state = state.transition(action)
   pass
