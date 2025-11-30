from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
from sctp import param, core


   

class GroundState(object):
    def __init__(self, graph=None, goalID=None, robot=None, iscopy=False, useOptHeur=True, n_maps=100):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.noway2goal = False
        self.depth = 0
        # self.max_depth = self.depth
        self.vertices_map = dict() # map vertex id to vertex object
        self.sampling_maps = n_maps
        self.actions = []
        self.robot = robot
        self.uavs = []
        self.going_back = False
        self.use_OptHeur = useOptHeur
        if not iscopy:
            self.graph = graph
            self.goalID = goalID
            self.history = core.History()
            # self.vertices_set = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.init_history()
            self.visited_vertices = dict()
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            # define robot
            self.robot = robot
            self.robot.need_action = True
            if self.robot.at_node:
                neighbors = [node for node in self.graph.vertices+self.graph.pois if node.id == robot.last_node][0].neighbors
                self.actions = [core.Action(target=neighbor, start_pose=robot.cur_pose) for neighbor in neighbors]
                self.visited_vertices[self.robot.last_node] = 1
                if self.history.get_action_outcome(core.Action(target=self.robot.last_node))==param.EventOutcome.BLOCK:
                    self.actions = [action for action in self.actions if action.target ==self.robot.pl_vertex]                
            else:
                self.actions = [core.Action(target=self.robot.edge[0], start_pose=self.robot.cur_pose), 
                                        core.Action(target=self.robot.edge[1],start_pose=self.robot.cur_pose)]
            self.actions = [action for action in self.actions \
                                if self.history.get_action_outcome(action) != core.EventOutcome.BLOCK]
            self.update_heuristic()
            # self.noway2goal = is_robot_stuck(self)
        assert self.uavs == []
    def init_history(self):
        for vertex in self.graph.vertices+self.graph.pois:
            action = core.Action(target=vertex.id)
            if vertex.block_prob == 1.0:
                self.history.add_history(action, param.EventOutcome.BLOCK)
            elif vertex.block_prob == 0.0:
                self.history.add_history(action, param.EventOutcome.TRAV)
                
    def get_actions(self):
        return self.actions

    def copy(self):
        new_state = GroundState(iscopy=True)
        new_state.depth = self.depth
        new_state.sampling_maps = self.sampling_maps
        new_state.going_back = self.going_back
        new_state.history = self.history.copy()
        new_state.graph = self.graph
        new_state.goalID = self.goalID
        new_state.action_cost = 0.0
        new_state.visited_vertices = self.visited_vertices.copy()
        new_state.noway2goal = self.noway2goal
        new_state.heuristic = self.heuristic
        new_state.robot = self.robot.copy()
        # new_state.uavs = [uav.copy() for uav in self.uavs]
        new_state.actions = [action for action in self.actions]
        new_state.vertices_map = self.vertices_map
        for action in new_state.actions:
            action.update_pose((new_state.robot.cur_pose[0], new_state.robot.cur_pose[1]))                
        return new_state

    def update_heuristic(self):
        # assert self.sampling_maps == 80
        self.noway2goal = False
        block_pois = [key.target for key, value in self.history.get_data().items() if value == param.EventOutcome.BLOCK]
        if self.robot.at_node:
            if self.robot.last_node in block_pois:
                block_pois.remove(self.robot.last_node)
            new_graph = g.remove_pois(graph=self.graph, poiIDs=block_pois)
            heuristic, _ = paths.get_shortestPath_cost(graph=new_graph, start=self.robot.last_node, goal=self.goalID)
            if heuristic < 0.0:
                self.heuristic = param.STUCK_COST
                self.noway2goal = True
            else:
                self.heuristic = heuristic
            return heuristic
            
        else:
            redge = [self.robot.edge[0], self.robot.edge[1]]
            new_pois = [p for p in block_pois if p != redge[0] and p != redge[1]]
            # new_graph = g.modify_graph(graph=self.graph, robot_edge=redge, poiIDs=block_pois)
            new_graph = g.remove_pois(graph=self.graph, poiIDs=new_pois)        
            min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalID)
            min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalID)
            # if min_dist1 < 0.0 or min_dist2 < 0.0:
            #     print(f"in Gstate_dect check the edge {redge} with dist1={min_dist1} and dist2={min_dist2}")
            #     print(f"inputed blocked pois: {block_pois} and new pois: {new_pois}")
            #     print(f"remained edges: {[{edge.v1.id, edge.v2.id} for edge in new_graph.edges]}")
            #     for v in self.graph.vertices+self.graph.pois:
            #         if v.id == redge[0] or v.id == redge[1]:
            #             print(f"In the input graph: Vertex {v.id} with block prob {v.block_prob} and neighbors {v.neighbors}")

                
            #     for v in new_graph.vertices+new_graph.pois:
            #         if v.id == redge[0] or v.id == redge[1]:
            #             print(f"In the modified graph: Vertex {v.id} with block prob {v.block_prob} and neighbors {v.neighbors}")

            assert (min_dist1 < 0.0) == (min_dist2 < 0.0)
            # print("in Gstate_dect  -- Satisfied assertion for min_dist1 and min_dist2")
            if min_dist1 < 0.0 and min_dist2 < 0.0:
                self.heuristic = param.STUCK_COST
                self.noway2goal = True
                return self.heuristic
            if self.use_OptHeur:
                self.heuristic = min(min_dist1, min_dist2)
            else:
                d2 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
                d1 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
                self.heuristic = core.sampling_rollout(new_graph, redge, d1, d2, self.goalID, self.robot.at_node, 
                                                startNode=self.robot.last_node, n_maps=self.sampling_maps)        
        return self.heuristic
    
    @property
    def is_goal_state(self):
        return (self.robot.last_node == self.goalID) or self.noway2goal

    @property
    def is_block_state(self):
        return self.noway2goal

    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

    def transition(self, action):
        temp_state = self.copy()
        assert temp_state.robot.need_action == True
        start_pos = (temp_state.robot.cur_pose[0],temp_state.robot.cur_pose[1])
        action.update_pose(start_pos)
        action.update_robotID(self.robot.id)
        distance, direction = temp_state.get_distance_direction(start_pos, action.target)
        temp_state.robot.retarget(action, distance, direction)
        return advance_state(temp_state)     

def advance_state(state):
    state.depth += 1
    if state.robot.last_node != state.goalID:
        state.robot.advance_time(state.robot.remaining_time)
    vertex_status = state.history.get_action_outcome(state.robot.action)
    vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.action.target][0]
    state.robot.visited_vertices.append(state.robot.last_node)
    state.visited_vertices[state.robot.last_node] = state.visited_vertices.get(state.robot.last_node, 0) + 1
    assert state.robot.at_node == True
    if vertex_status == param.EventOutcome.BLOCK:
        state.actions = [core.Action(target=state.robot.pl_vertex, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV: 
        cur_node = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.last_node][0]
        state.actions = [core.Action(target=neighbor, start_pose=(state.robot.cur_pose[0], \
                        state.robot.cur_pose[1])) for neighbor in cur_node.neighbors if neighbor !=state.robot.pl_vertex]
        state.actions = [action for action in state.actions \
                                    if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
        # state.noway2goal = is_robot_stuck(state)
        # update the cost if revisiting the vertex
        state.action_cost += (state.visited_vertices.get(state.robot.last_node, 0)-1) * param.REVISIT_PEN
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        new_state_trav = get_new_robot_node(state)
        new_state_block = get_new_robot_node(state, blocked=True)
        assert new_state_block.depth == new_state_trav.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}
        
def get_new_robot_node(state, blocked=False):
    new_state = state.copy()
    new_state.action_cost = state.action_cost
    if blocked:
        new_state.going_back = True
        new_state.history.add_history(state.robot.action, param.EventOutcome.BLOCK)
        new_state.actions = [core.Action(target=new_state.robot.pl_vertex, \
                                        start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
    else:
        new_state.going_back = False
        new_state.history.add_history(state.robot.action, param.EventOutcome.TRAV)
        neighbors = [node for node in state.graph.pois if node.id == state.robot.last_node][0].neighbors
        new_state.actions = [core.Action(target=neighbor,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
                                    for neighbor in neighbors if neighbor != state.robot.pl_vertex]
    # new_state.noway2goal = is_robot_stuck(new_state)
    new_state.update_heuristic()
    return new_state

def is_robot_stuck(state):
    if state.robot.last_node == state.goalID:
        state.noway2goal = False
        return False
    if state.robot.at_node:
        robot_edge = [state.robot.last_node, state.robot.pl_vertex]
    else:
        robot_edge = [state.robot.edge[0],state.robot.edge[1]]
    if (not core._is_robot_goal_connected(state.graph, state.history, robot_edge, state.goalID))\
        or (len(state.actions) == 0):
        state.noway2goal = True
        return True
    return False
