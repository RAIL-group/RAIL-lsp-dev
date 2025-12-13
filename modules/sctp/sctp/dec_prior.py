from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
import random
from sctp import param, core
from sctp import action_esti as ae
import pytest
from sctp.gstate_dec import GroundState 
import pouct_planner

# policy = {}
# policy[(history, node)] = actionID

class StateDecPrior(object):
    def __init__(self, graph=None, goalIDs=[], ugvs=[], drones=[], 
                 iscopy=False, n_maps=100, use2AG=False, max_uanum=3):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.depth = 0
        self.vertices_map = dict() # map vertex id to vertex object
        self.sampling_maps = n_maps
        self.going_back = False
        self.state_actions = []
        self.use_OptiHeur = True
        self.ugvs_policies = [] # list of ugpolicy
        self.noway2goal = False
        self.cur_ugv_idx = -1
        self.use_2AG = use2AG
        self.max_uanum = max_uanum
        
        if not iscopy:
            # need to filter the visited POIs
            assert graph is not None
            assert ugvs != []
            self.graph = graph
            self.goalIDs = goalIDs
            self.history = core.History()
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.assigned_pois = set()
            self.init_history()
            self.action_values = dict() # map action to its value
            self.behavior_change = dict() # map action to its value
            self.ugvs = ugvs                            
            # continue_actions = []
            self.ugvs_actions = []
            for i, ugv in enumerate(self.ugvs):
                ugv.need_action = True
                if ugv.last_node != self.goalIDs[i]:
                        ugv_actions = [get_ugv_action(self, i)]
                else:
                    ugv_actions = [core.Action(target=self.goalIDs[i], start_pose=ugv.cur_pose)]
                ugv.visited_vertices.append(ugv.last_node)
                # if ugv.at_node:
                    # neighbors = [node for node in self.graph.vertices+self.graph.pois if node.id == ugv.last_node][0].neighbors
                    # ugv_actions = [core.Action(target=neighbor, start_pose=ugv.cur_pose) for neighbor in neighbors]
                    # ugv.visited_vertices.append(ugv.last_node)
                    # if self.history.get_action_outcome(core.Action(target=ugv.last_node))==param.EventOutcome.BLOCK:
                    #     ugv_actions = [action for action in ugv_actions if action.target ==ugv.pl_vertex]                
                    # if ugv.last_node != self.goalIDs[i]:
                    #     ugv_actions = [get_ugv_action(self, i)]
                    # else:
                    #     ugv_actions = [core.Action(target=self.goalIDs[i], start_pose=ugv.cur_pose)]
                # else:
                #     ugv_actions = [core.Action(target=ugv.edge[0], start_pose=ugv.cur_pose), 
                #                           core.Action(target=ugv.edge[1],start_pose=ugv.cur_pose)]
                # ugv_actions = [action for action in ugv_actions \
                #                   if self.history.get_action_outcome(action) != core.EventOutcome.BLOCK]
                assert len(ugv_actions) > 0
                self.ugvs_actions.append(ugv_actions)
            self.cur_ugv_idx = 0
            self.state_actions = [action for action in self.ugvs_actions[0]] # what if this ugv has reached goal?
            self.update_heuristic()
            
            self.uavs = drones
            if len(self.uavs) > 0:
                self.uav_actions = [core.Action(target=poi.id, rtype=param.RobotType.Drone) for poi in self.graph.pois]                
                self.uav_actions = [action for action in self.uav_actions \
                                    if self.history.get_action_outcome(action) == param.EventOutcome.CHANCE] # list unexplored pois
                # continue_actions = []
                for i, uav in enumerate(self.uavs):
                    # assert uav.unfinished_action is None
                    if uav.unfinished_action and uav.unfinished_action in self.uav_actions:
                        uav.action = uav.unfinished_action
                        # continue_actions.append(uav.action)
                        assert uav.action.rtype == param.RobotType.Drone
                        uav.unfinished_action = None
                        uav.need_action = False
                        self.assigned_pois.add(uav.action.target)
                        uav.action.update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                        uav.action.update_robotID(i)
                        distance, direction = self.get_distance_direction(uav.cur_pose, uav.action.target)
                        uav.retarget(uav.action, distance, direction)
                        self.uav_actions.remove(uav.action)
                    else:
                        uav.need_action = True
                        uav.action = None
                
                indices = [i for i, uav in enumerate(self.uavs) if uav.need_action]
                if self.use_2AG:
                    self.behavior_change.clear()
                    self.action_values.clear()
                    # get the index of uavs  needing action index
                    if indices != []:
                        uav_idx = indices[0]
                        for act in self.uav_actions: # only for value information gain 118-127
                            # if act in continue_actions:
                            if act.target in self.assigned_pois:
                                continue
                            self.behavior_change[act] = ae.get_ugvs_behavior_change(state=self, action=act)
                            self.action_values[act] = ae.get_action_value(self.behavior_change[act], act, 
                                                                    self.uavs[uav_idx].cur_pose, self.graph)
                        self.action_values = dict(sorted(self.action_values.items(), key=lambda item: item[1], reverse=True))
                        self.uav_actions = list(self.action_values.keys())[:min(self.max_uanum, len(self.action_values))]
                        for action in self.uav_actions:
                            action.update_pose((self.uavs[uav_idx].cur_pose[0], self.uavs[uav_idx].cur_pose[1]))
                            action.update_robotID(uav_idx)
                        assert len(self.uav_actions) <= self.max_uanum
                        for _ in range(min(self.max_uanum, len(self.action_values))):
                            first_key = next(iter(self.action_values))
                            self.action_values.pop(first_key)
                            self.behavior_change.pop(first_key, None)
                         
                if len(self.uav_actions) == 0:
                    self.uav_actions = [core.Action(target=self.goalIDs[0], rtype=param.RobotType.Drone)]
                    if indices != []:
                        self.uav_actions[0].update_pose((self.uavs[indices[0]].cur_pose[0], self.uavs[indices[0]].cur_pose[1]))
                        self.uav_actions[0].update_robotID(indices[0])      
                    else:
                        self.uav_actions[0].update_pose((self.uavs[0].cur_pose[0], self.uavs[0].cur_pose[1]))
                        self.uav_actions[0].update_robotID(0)
                # check right here
                assert isinstance(self.uav_actions[0], core.Action)
                assert len(self.uav_actions) > 0
                if any([uav.need_action for uav in self.uavs]):
                    self.cur_ugv_idx = -1
                    self.state_actions = [action for action in self.uav_actions]
            
    def get_actions(self):
        return self.state_actions
    
    def init_history(self):
        for vertex in self.graph.vertices+self.graph.pois:
            action = core.Action(target=vertex.id)
            if vertex.block_prob == 1.0:
                self.history.add_history(action, param.EventOutcome.BLOCK)
            elif vertex.block_prob == 0.0:
                self.history.add_history(action, param.EventOutcome.TRAV)
                

    def update_heuristic(self):
        block_pois = [key.target for key, value in self.history.get_data().items() if value == core.EventOutcome.BLOCK]
        edges, vertices_wrobot = g.get_removedEdges_allrobots(self.graph, self.ugvs, block_pois)
        new_pois = [poi for poi in block_pois if poi not in vertices_wrobot]        
        
        new_graph = g.remove_pois(self.graph, new_pois)
        new_graph = g.remove_edges(new_graph, edges)
        self.heuristic = 0.0
        assert self.use_OptiHeur == True
        for i, robot in enumerate(self.ugvs):
            heuristic_cost = 0.0
            if robot.at_node:
                redge = [robot.last_node, robot.last_node]
                if self.use_OptiHeur:
                    heuristic_cost, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalIDs[i])
                    if heuristic_cost < 0.0:
                        self.noway2goal = True
                else:
                    assert 1 == 0
                    heuristic_cost = core.sampling_rollout(new_graph, [robot.last_node,robot.last_node], 0.0, 0.0, self.goalIDs[i], 
                                                robot.at_node, startNode=robot.last_node, n_samples=self.n_maps)
            else:
                redge = [robot.edge[0], robot.edge[1]]            
                d1 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
                d2 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
                if self.use_OptiHeur:
                    min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalIDs[i])
                    min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalIDs[i])
                    assert (min_dist1 < 0) == (min_dist2 < 0)
                    if min_dist1 < 0.0 and min_dist2 < 0.0:
                        heuristic_cost = -1.0
                        self.noway2goal = True
                    else:
                        heuristic_cost = min(d1 + min_dist1, d2 + min_dist2)
                else:
                    assert 1 == 0
                    heuristic_cost = core.sampling_rollout(new_graph, redge, d1, d2, self.goalIDs[i], robot.at_node,
                                                startNode=robot.last_node, n_samples=self.n_maps)
            heuristic_cost = param.NOWAY_PEN if heuristic_cost < 0.0 else heuristic_cost
            self.heuristic += heuristic_cost                   
        return self.heuristic
    
    def update_action_values(self, uav_idx):
        self.action_values.clear()
        for action, value in self.behavior_change.items():
            action_value = ae.get_action_value(bc=value, action=action, drone_pose=self.uavs[uav_idx].cur_pose, graph=self.graph)
            self.action_values[action] = action_value
    
    def update_action_bc(self):
        self.behavior_change.clear()
        curr_actions = [core.Action(target=poi.id, rtype=param.RobotType.Drone) for poi in self.graph.pois \
                        if poi.id not in self.assigned_pois]                
        curr_actions = [action for action in curr_actions \
                        if self.history.get_action_outcome(action) == param.EventOutcome.CHANCE] 
        for ii, act in enumerate(curr_actions):
           self.behavior_change[act] = ae.get_ugvs_behavior_change(state=self, action=act)
                
    @property
    def is_goal_state(self):
        return all ([ugv.last_node == self.goalIDs[i] for i, ugv in enumerate(self.ugvs)])

    @property
    def is_block_state(self):
        return self.noway2goal

    def copy(self):
        new_state = StateDecPrior(iscopy=True)
        new_state.cur_ugv_idx = self.cur_ugv_idx
        new_state.heuristic = self.heuristic
        new_state.vertices_map = self.vertices_map.copy()
        new_state.depth = self.depth
        new_state.graph = self.graph
        new_state.sampling_maps = self.sampling_maps
        new_state.goalIDs = self.goalIDs.copy()
        new_state.going_back = False
        new_state.use_OptiHeur = self.use_OptiHeur
        new_state.use_2AG = self.use_2AG
        new_state.max_uanum = self.max_uanum
        new_state.action_cost = 0.0
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.history = self.history.copy()
        if self.use_2AG:
            new_state.behavior_change = self.behavior_change.copy()
            new_state.action_values = self.action_values.copy()
        else:
            new_state.behavior_change = dict()
            new_state.action_values = dict()
            
        new_state.state_actions = []
        # copy the robot
        new_state.ugvs = [ugv.copy() for ugv in self.ugvs]
        new_state.ugvs_actions = [[core.Action(target=action.target, start_pose=(action.start_pose[0],action.start_pose[1])) \
                                    for action in ugv_actions] for ugv_actions in self.ugvs_actions]
        for ii, actions in enumerate(new_state.ugvs_actions):
            for act in actions:
                act.update_robotID(ii)
        if self.uavs != []:
            new_state.uavs = [uav.copy() for uav in self.uavs]
            if self.use_2AG:
                new_state.uav_actions = []
            else:
                new_state.uav_actions = [core.Action(target=action.target, rtype=action.rtype) \
                                    for action in self.uav_actions]
                # for i, uav in enumerate(new_state.uavs):
                #     if uav.need_action and uav.last_node != new_state.goalIDs[0]:
                #         new_state.uav_actions[0].update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                #         new_state.uav_actions[0].update_robotID(i)
        else:
            new_state.uavs = []
            new_state.uav_actions = []
        return new_state
        
    def transition(self, action):
        temp_state = self.copy()
        anyrobot_action = False        
        if action.rtype == param.RobotType.Drone:
            uav_needs_action = [uav.need_action for uav in temp_state.uavs]
            assert any(uav_needs_action) == True
            if self.use_2AG:
                uav_idx = action.robotID
            else:
                uav_idx = uav_needs_action.index(True)
                # if not self.use_2AG:
                assert action in temp_state.uav_actions
            assert temp_state.cur_ugv_idx == -1
            if action.target in temp_state.assigned_pois:
                print(f"Assigning action {action.target} to drone {uav_idx} that is in the assigned_pois {temp_state.assigned_pois}")
            assert action.target not in temp_state.assigned_pois # related to add new action into assigned_pois
            
            start_pos = (temp_state.uavs[uav_idx].cur_pose[0], temp_state.uavs[uav_idx].cur_pose[1])
            if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
                ValueError("Start position is NaN") 
            action.update_pose(start_pos)
            action.update_robotID(uav_idx)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)            
            temp_state.uavs[uav_idx].retarget(action, distance, direction)
            if action.target != temp_state.goalIDs[0]:
                temp_state.assigned_pois.add(action.target)
                if not self.use_2AG:
                    temp_state.uav_actions.remove(action)
            # using action estimation
            if self.use_2AG:
                temp_state.behavior_change.pop(action, None)
                temp_state.action_values.pop(action, None)
                if any ([uav.need_action for uav in temp_state.uavs]):
                    self.action_values.clear()
                    n_uav_idx = [i for i, uav in enumerate(temp_state.uavs) if uav.need_action][0]
                    temp_state.uav_actions = ae.get_uav_action_2ag(temp_state, n_uav_idx)
            
            if len(temp_state.uav_actions) ==0:
                rest_action = core.Action(target=temp_state.goalIDs[0], rtype=param.RobotType.Drone)
                temp_state.uav_actions = [rest_action]
                if self.use_2AG: 
                    if any ([uav.need_action for uav in temp_state.uavs]):
                        rest_action.update_pose((temp_state.uavs[n_uav_idx].cur_pose[0], temp_state.uavs[n_uav_idx].cur_pose[1]))
                        rest_action.update_robotID(n_uav_idx)
                    else:
                        rest_action.update_pose((temp_state.uavs[0].cur_pose[0], temp_state.uavs[0].cur_pose[1]))
                        rest_action.update_robotID(0)
                
            anyrobot_action = True
        elif action.rtype == param.RobotType.Ground:
            assert temp_state.cur_ugv_idx > -1
            ugv_needs_action = [ugv.need_action for ugv in temp_state.ugvs]
            assert any(ugv_needs_action) == True
            # ugv_idx = ugv_needs_action.index(True)
            ugv_idx = temp_state.cur_ugv_idx
            start_pos = (temp_state.ugvs[ugv_idx].cur_pose[0], temp_state.ugvs[ugv_idx].cur_pose[1])
            action.update_pose(start_pos)
            action.update_robotID(ugv_idx)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)
            temp_state.ugvs[ugv_idx].retarget(action, distance, direction)
            anyrobot_action = True
        assert anyrobot_action == True
        return advance_state(temp_state, action)

    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

def get_ugv_action(state, ugv_idx):
    # we can get the action from set of policies such as
    # return action = policy[ugv_idx][(state.history, state.ugvs[ugv_idx].last_node)]
    # but it is for the next step,
    # right now we just compute it online
    return update_policy(state, ugv_idx)
    
    
def update_policy(state, ugv_idx, num_rollouts=1000):
    C = 200.0
    max_depth = 10
    useOptHeur = True
    revisit_pen = 20.0
    
    new_graph  = state.graph.copy()
    for poi in new_graph.pois:
        if (state.history.get_action_outcome(core.Action(target=poi.id)) == param.EventOutcome.BLOCK):
            poi.block_prob = 1.0
        elif (state.history.get_action_outcome(core.Action(target=poi.id)) == param.EventOutcome.TRAV):
            poi.block_prob = 0.0
    robot = state.ugvs[ugv_idx].copy()
    sctpstate = GroundState(graph=new_graph, goalID=state.goalIDs[ugv_idx], robot=robot, \
                            useOptHeur=useOptHeur, n_maps=state.sampling_maps, revisit_pen=revisit_pen)
    
    action, _, [ordering, costs] = pouct_planner.core.po_mcts(sctpstate, \
                    n_iterations=num_rollouts, C=C, depth= max_depth, \
                    rollout_fn=decsctp_rollout)
    return action

def advance_state(state, action):
    assert state.going_back == False
    # 1. if any robot needs action, determine its actions then return
    if state.uavs != [] and any([uav.need_action for uav in state.uavs]) and action.rtype == param.RobotType.Drone:
        state.state_actions = [action for action in state.uav_actions]
        state.cur_ugv_idx = -1
        if len(state.state_actions) == 0:
            for robot in state.uavs+state.ugvs:
                print(f"{robot.robot_type} {robot.id} need_action: {robot.need_action} is atNode?? {robot.at_node} at node {robot.last_node} at {robot.cur_pose} from {robot.pl_vertex}" )
            ValueError("No available action for UAVs")
            
        assert len(state.state_actions) > 0
        state.depth += 1
        return {state: (1.0, 0.0)}
    robots_need_action = [robot.need_action for robot in state.ugvs]
    if any(robots_need_action):
        # set action for this state        
        robot_idx = robots_need_action.index(True)
        state.ugvs_actions[robot_idx] = [action for action in state.ugvs_actions[robot_idx] \
                            if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
        if len(state.ugvs_actions[robot_idx]) == 0: # pre-assigned action is blocked.
            state.ugvs_actions[robot_idx] = [core.Action(target=state.ugvs[robot_idx].pl_vertex, \
                            start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
            state.ugvs_actions[robot_idx] = [action for action in state.ugvs_actions[robot_idx] \
                            if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
            for action in state.ugvs_actions[robot_idx]:
                action.update_robotID(robot_idx)
        if len(state.ugvs_actions[robot_idx]) == 0:
            state.noway2goal = True
            state.state_actions = []
        else:
            state.state_actions = [action for action in state.ugvs_actions[robot_idx]]
            assert len(state.state_actions) > 0
        state.cur_ugv_idx = robot_idx
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, rd_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_nodes = [robot.last_node for robot in state.ugvs]
    edges = [robot.edge for robot in state.ugvs]
    # move the robots
    for i, robot in enumerate(state.ugvs):
        if robot.last_node != state.goalIDs[i]:
            robot.advance_time(time_advance)
        else:
            robot.need_action = False
    for uav in state.uavs:
        if uav.last_node != state.goalIDs[0]:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
    if robot_reach_first:
        return get_ugv_belief(state, last_nodes=last_nodes, robot_idx = rd_index, last_edges=edges)
    else:
        return get_uav_belief(state=state, uav_index=rd_index)
    

def get_ugv_belief(state, last_nodes, robot_idx, last_edges): # need to work on this.
    vertex_status = state.history.get_action_outcome(state.ugvs[robot_idx].action)
    vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.ugvs[robot_idx].action.target][0]
    state.ugvs[robot_idx].visited_vertices.append(state.ugvs[robot_idx].last_node)
    # state.visited_vertices[state.ugvs[robot_idx].last_node] = state.visited_vertices.get(state.ugvs[robot_idx].last_node, 0) + 1
    assert state.ugvs[robot_idx].at_node == True
    state.cur_ugv_idx = robot_idx
    if vertex_status == param.EventOutcome.BLOCK:
        state.ugvs_actions[robot_idx] = [core.Action(target=state.ugvs[robot_idx].pl_vertex, \
                            start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
        for action in state.ugvs_actions[robot_idx]:
            action.update_robotID(robot_idx)
        state.state_actions = [action for action in state.ugvs_actions[robot_idx]]
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV: 
        if state.ugvs[robot_idx].last_node == state.goalIDs[robot_idx]:
            state.ugvs_actions[robot_idx] = [core.Action(target=state.goalIDs[robot_idx], \
                        start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
        else:
            neighbors = [nei for nei in state.vertices_map[state.ugvs[robot_idx].last_node].neighbors]
            if (state.ugvs[robot_idx].pl_vertex != state.ugvs[robot_idx].last_node) and state.ugvs[robot_idx].pl_vertex in neighbors:
                    neighbors.remove(state.ugvs[robot_idx].pl_vertex)
            if len(neighbors) > 1:
                state.ugvs_actions[robot_idx] = [get_ugv_action(state, robot_idx)]                
            elif len(neighbors) == 1:
                state.ugvs_actions[robot_idx] = [core.Action(target=neighbors[0], \
                    start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1]))]
            else:
                state.noway2goal = True
                state.ugvs_actions[robot_idx] = []
        for action in state.ugvs_actions[robot_idx]:
            action.update_robotID(robot_idx)
        state.state_actions = [action for action in state.ugvs_actions[robot_idx] ]
        state.update_heuristic()
        if state.use_2AG:
            state.update_action_bc()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        if len(state.uavs) > 0:
            reset_uavs_action(state, robot_idx)
        # update the uav action set.
        for i, ugv in enumerate(state.ugvs):
            if i != robot_idx:
                ugv.need_action = False
        # TRAVERSABLE
        new_state_trav = get_new_ugv_node(state, robot_idx=robot_idx)
        new_state_block = get_new_ugv_node(state, robot_idx=robot_idx, last_node=last_nodes[robot_idx], blocked=True)
        assert new_state_block.depth == new_state_trav.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}
        
def get_new_ugv_node(state, robot_idx, last_node=None, blocked=False):
    new_state = state.copy()
    new_state.current_ugv_idx = robot_idx
    new_state.action_cost = state.action_cost
    if blocked:
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.BLOCK)
        assert last_node is not None
        if last_node != new_state.ugvs[robot_idx].last_node:
            target = last_node
        else:
            target = new_state.ugvs[robot_idx].pl_vertex
        new_state.ugvs_actions[robot_idx] = [core.Action(target=target, \
                    start_pose=(state.ugvs[robot_idx].cur_pose[0], state.ugvs[robot_idx].cur_pose[1]))]
        new_state.state_actions = [action for action in new_state.ugvs_actions[robot_idx]]
    else:
        new_state.history.add_history(state.ugvs[robot_idx].action, param.EventOutcome.TRAV)
        neighbors = [node for node in state.graph.vertices+state.graph.pois if node.id == state.ugvs[robot_idx].last_node][0].neighbors
        new_state.ugvs_actions[robot_idx] = [core.Action(target=neighbor, \
                                    start_pose=(state.ugvs[robot_idx].cur_pose[0],state.ugvs[robot_idx].cur_pose[1])) \
                                    for neighbor in neighbors if neighbor != state.ugvs[robot_idx].pl_vertex]        
        new_state.state_actions = [action for action in new_state.ugvs_actions[robot_idx]]
    for action in new_state.ugvs_actions[robot_idx]:
        action.update_robotID(robot_idx)
    for i, robot in enumerate(new_state.ugvs):
        if i != robot_idx and not robot.at_node: # reset if it is in middle of action
            robot.need_action = True
            robot.remaining_time = 0.0
            action = get_ugv_action(new_state, i)
            action.update_pose((robot.cur_pose[0], robot.cur_pose[1]))
            action.update_robotID(i)
            new_state.ugvs_actions[i] = [action]
    new_state.update_heuristic()
    if new_state.use_2AG:
        new_state.update_action_bc()
    else:
        new_state.uav_actions = [action for action in state.uav_actions if action.target != state.ugvs[robot_idx].last_node]
    
    
    return new_state

def get_uav_belief(state, uav_index):
    assert len(state.uavs) > 0
    vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
    vertex = [node for node in state.graph.pois+state.graph.vertices if node.id == state.uavs[uav_index].action.target][0]
    # determine all actions related to the poi and remove it from the set.
    poi_id = state.uavs[uav_index].last_node
    assert poi_id == vertex.id
    # state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
    # if len(state.uav_actions) == 0:
    #     if state.use_2AG:
    #         state.update_action_bc()
    #         state.update_action_values(uav_index)
    #         state.uav_actions = list(state.action_values.keys())[:min(state.max_uanum, len(state.action_values))]
    #     else:
            # rest_action = core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)
            # state.uav_actions.append(rest_action)
    # if some uavs also finish theirs actions, reset need_action for the next iteration
    for i, uav in enumerate(state.uavs):
        uav.need_action = False if i != uav_index and uav.need_action else uav.need_action
    # just assuming ugvs do not need actions now
    for i, robot in enumerate(state.ugvs):
        robot.need_action = False
    state.cur_ugv_idx = -1
    if vertex_status == param.EventOutcome.BLOCK: # should not go here
        # ValueError("Drones should never visit this node")
        state.depth += 1
        if state.use_2AG:
            state.update_action_bc()
            state.update_action_values(uav_index)
            state.uav_actions = ae.get_uav_action_2ag(state, uav_index)
        else:
            state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV:  # only at goal
        state.depth += 1
        if vertex.id == state.goalIDs[0]:
            state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
            state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
            state.uav_actions[0].update_robotID(uav_index)
        else:
            if state.use_2AG:
                state.update_action_bc()
                state.update_action_values(uav_index)
                state.uav_actions = ae.get_uav_action_2ag(state, uav_index)
            else:
                state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
            if len(state.uav_actions) == 0:
                state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
                if state.use_2AG:
                    state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
                    state.uav_actions[0].update_robotID(uav_index)
        state.state_actions = [action for action in state.uav_actions]
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        new_state_trav = get_new_uav_node(state, uav_index, blocked=False) # TRAVERSABLE
        new_state_block = get_new_uav_node(state, uav_index, blocked=True) # BLOCKED
        assert new_state_trav.depth == new_state_block.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def get_new_uav_node(state, uav_index, blocked=False):
    new_state = state.copy()
    new_state.cur_ugv_idx = -1
    new_state.action_cost = state.action_cost
    if blocked:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.BLOCK)
    else:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.TRAV)
    for i, robot in enumerate(new_state.ugvs):
        if not robot.at_node: # reset if it is in middle of action
            robot.need_action = True # you don't want it to go to get_new_nodes_grobot (no node reached)
            robot.remaining_time = 0.0
            action = get_ugv_action(new_state, i)            
            action.update_pose((robot.cur_pose[0], robot.cur_pose[1]))
            new_state.ugvs_actions[i] = [action]
    new_state.update_heuristic()
    assert new_state.use_2AG == state.use_2AG
    assert new_state.max_uanum == state.max_uanum
    if new_state.use_2AG:
        new_state.update_action_bc()
        new_state.update_action_values(uav_index)
        new_state.uav_actions = ae.get_uav_action_2ag(new_state, uav_index)
    if len(new_state.uav_actions)  == 0:
        state.uav_actions = [core.Action(target=state.goalIDs[0], rtype=param.RobotType.Drone)]
        if state.use_2AG:
            state.uav_actions[0].update_pose((state.uavs[uav_index].cur_pose[0], state.uavs[uav_index].cur_pose[1]))
            state.uav_actions[0].update_robotID(uav_index)
    new_state.state_actions = [action for action in new_state.uav_actions]
    return new_state


def reset_uavs_action(state, robot_idx):
    for i, uav in enumerate(state.uavs):
        if not uav.need_action and uav.action.target == state.ugvs[robot_idx].action.target \
                    and uav.action.target != state.goalIDs[0]:
            uav.need_action = True 
            uav.remaining_time = 0.0

def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    ugv_finish_first = True
    if len(state.uavs) > 0:
        for uav in state.uavs:
            if uav.last_node == state.goalIDs[0] and uav.remaining_time <= param.APPROX_TIME:
                continue
            time_remaining_uavs.append(uav.remaining_time)
    ugvs_remaining_times = []
    for i, ugv in enumerate(state.ugvs):
        if ugv.last_node == state.goalIDs[i] and ugv.remaining_time <= param.APPROX_TIME:
            continue
        ugvs_remaining_times.append(ugv.remaining_time)
    assert len(ugvs_remaining_times) > 0
    min_ugv_time = min(ugvs_remaining_times)
    if len(time_remaining_uavs)==0 or (min_ugv_time < min(time_remaining_uavs)-param.APPROX_TIME):
        remaining_times = [ugv.remaining_time for ugv in state.ugvs]
        idx = -1
        for ii, time in enumerate(remaining_times):
            if state.ugvs[ii].last_node != state.goalIDs[ii] and time ==min_ugv_time:
                idx = ii
        assert idx >= 0
        return ugv_finish_first, idx, min_ugv_time # remaining_times.index(min_ugv_time), min_ugv_time
    else:
        assert len(state.uavs) > 0
        min_uav_time = min(time_remaining_uavs)
        remaining_times = [uav.remaining_time for uav in state.uavs]
        idx = -1
        for ii, time in enumerate(remaining_times):
            if state.uavs[ii].last_node != state.goalIDs[0] and time ==min_uav_time:
                idx = ii
        assert idx >= 0
        ugv_finish_first = False
        return ugv_finish_first, idx, min_uav_time #remaining_times.index(min_uav_time), min_uav_time


def decsctp_rollout(state):
    if state.is_goal_state and not state.is_block_state:
        return 0.0
    if state.heuristic >= 0.0:
        return state.heuristic
    return state.update_heuristic()

