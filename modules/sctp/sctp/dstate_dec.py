from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
from sctp import param, core
import random
    

class DronesState(object):
    def __init__(self, actions=None, robot_pos=None, redges=None, graph=None, 
                 restID=None, drones=[], iscopy=False, n_maps=100):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.depth = 0
        self.vertices_map = dict() # map vertex id to vertex object
        self.n_samples = n_maps
        self.action_values = dict() # map action to its value
        self.behavior_change = dict() # map action to its value
        self.going_back = False
        self.sampling_time = 0.0 # no using in this state
        self.s_policy_time = 0.0 # no using in this state
        
        if not iscopy:
            assert actions is not None
            assert robot_pos is not None
            assert redges is not None
            # need to filter the visited POIs
            self.grobot_pos = robot_pos
            self.gredges = redges
            self.graph = graph
            self.restID = restID
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.actions = [action for action in actions if 0.0 < self.vertices_map[action.target].block_prob <1.0]
            self.assigned_pois = set()
            self.v_vertices = dict()
            self.heuristic_vertices = dict()
            # self.update_heuristic2()
            self.uavs = drones                             
            # continue_actions = []
            for i, uav in enumerate(self.uavs):
                # assert uav.unfinished_action is None
                if uav.unfinished_action and uav.unfinished_action in self.actions:
                    uav.action = uav.unfinished_action
                    self.assigned_pois.add(uav.action.target)
                    uav.action.update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                    uav.action.update_robotID(i)
                    # continue_actions.append(uav.unfinished_action)
                    uav.unfinished_action = None
                    uav.need_action = False
                    distance, direction = self.get_distance_direction(uav.cur_pose, uav.action.target)
                    uav.action.update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                    uav.action.update_robotID(i)
                    uav.retarget(uav.action, distance, direction)
                    self.actions.remove(uav.action)
                else:
                    uav.need_action = True
            if all([not uav.need_action for uav in self.uavs]):
                # print("All UAVs do not need action (having ongoing action), just advance!")
                advance_state(self, self.uavs[0].action)
            # if param.ADD_IV: # need to recompute action values - for all ground robots -------------------------
            #     for ii, act in enumerate(self.actions):
            #         bc = 0
            #         new_value = 0
            #         for jj, uav in enumerate(self.uavs):
            #             if act in continue_actions:
            #                 continue
            #             if self.gredges[jj][0] == self.gredges[jj][1]:
            #                 d1 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][0]].coord))
            #                 d2 = d1
            #                 isAtNode = True
            #             else:
            #                 d1 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][0]].coord))
            #                 d2 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][1]].coord))
            #                 isAtNode = False
            #             bc1 = core.get_behavior_change(graph=self.graph, action=act, robot_edge=self.gredges[jj],
            #                                         d0=d1, d1=d2, goalID=self.restID, atNode=isAtNode,
            #                                         cur_heuristic=self.heuristic, n_samples=param.IV_SAMPLE_SIZE)
            #             bc += bc1
            #             # new_value += core.get_action_value(bc=bc1, action=act, drone_pose=self.uavs[jj].cur_pose, graph=self.graph)
            #         self.behavior_change[act] = bc
            #         self.action_values[act] = bc
                    
            #     self.action_values = dict(sorted(self.action_values.items(), key=lambda item: item[1], reverse=True))
            #     self.actions = list(self.action_values.keys())[:min(param.MAX_UAV_ACTION, len(self.action_values))]
            #     assert len(self.actions) <= param.MAX_UAV_ACTION
            #     for _ in range(min(param.MAX_UAV_ACTION, len(self.action_values))):
            #         first_key = next(iter(self.action_values))
            #         self.action_values.pop(first_key)
                
            # if len(self.actions) < len(self.uavs):
            #     rest_action = core.Action(target=self.restID, rtype=param.RobotType.Drone)
            #     for i in range(len(self.uavs)-len(self.actions)):
            #         self.actions.append(rest_action)
            if len(self.actions) ==0:
                rest_action = core.Action(target=self.restID, rtype=param.RobotType.Drone)
                self.actions.append(rest_action)
                    
            assert isinstance(self.actions[0], core.Action)
            assert len(self.actions) > 0
            
    def get_actions(self):
        return self.actions
    
    # def init_history(self):
    #     for vertex in self.graph.vertices+self.graph.pois:
    #         action = core.Action(target=vertex.id)
    #         if vertex.block_prob == 1.0:
    #             self.history.add_history(action, param.EventOutcome.BLOCK)
    #         elif vertex.block_prob == 0.0:
    #             self.history.add_history(action, param.EventOutcome.TRAV)
                

    def update_heuristic2(self):
        pass
        #     redge = [self.robot.last_node, self.robot.pl_vertex]
        #     block_pois = [key.target for key, value in self.history.get_data().items() if value == param.EventOutcome.BLOCK]
        #     new_graph = g.modify_graph(graph=self.graph, robot_edge=redge, poiIDs=block_pois)        
        #     min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalID)
        #     min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalID)
        #     assert (min_dist1 < 0) == (min_dist2 < 0)
        #     if min_dist1 < 0.0 and min_dist2 < 0.0:
        #         self.heuristic = param.STUCK_COST
        #         return self.heuristic
        #     d2 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
        #     d1 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
        #     self.heuristic = core.sampling_rollout(new_graph, redge, d1, d2, self.goalID, self.robot.at_node, 
        #                                       startNode=self.robot.last_node, n_maps=self.n_samples)        
        #     return self.heuristic
        
    def update_action_value(self, uav_idx):
        self.action_values.clear()
        for action, value in self.behavior_change.items():
            action_value = core.get_action_value(bc=value, action=action, drone_pose=self.uavs[uav_idx].cur_pose, graph=self.graph)
            self.action_values[action] = action_value
    
    def update_action_bc(self):
        pass
        # block_pois = [key.target for key, value in self.history.get_data().items() if value == param.EventOutcome.BLOCK]
        # # // need a new function for this.
        # new_graph = g.modify_graph_multiDrones(graph=self.graph, robot_edges=self.gredges, poiIDs=block_pois)
        # sum_dist = 0.0
        # for jj, edge in enumerate(self.gredges):
        #     if edge[0] == edge[1]:
        #         min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=edge[0], goal=self.goalID)
        #     else:
        #         min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=edge[1], goal=self.goalID)
        #         min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=edge[0], goal=self.goalID)
        #         assert (min_dist1 < 0) == (min_dist2 < 0)
        #         min_dist1 = min(min_dist1, min_dist2)
        #     sum_dist += min_dist1
        
        # for ii, act in enumerate(self.actions):
        #     bc = 0
        #     new_value = 0
        #     for jj, uav in enumerate(self.uavs):
        #         if self.gredges[jj][0] == self.gredges[jj][1]:
        #             d1 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][0]].coord))
        #             d2 = d1
        #             isAtNode = True
        #         else:
        #             d1 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][0]].coord))
        #             d2 = np.linalg.norm(np.array(self.grobot_pos[jj])-np.array(self.vertices_map[self.gredges[jj][1]].coord))
        #             isAtNode = False
        #         bc1 = core.get_behavior_change(graph=self.graph, action=act, robot_edge=self.gredges[jj],
        #                                     d0=d1, d1=d2, goalID=self.restID, atNode=isAtNode,
        #                                     cur_heuristic=self.heuristic, n_samples=param.IV_SAMPLE_SIZE)
        #         bc += bc1
        #         # new_value += core.get_action_value(bc=bc1, action=act, drone_pose=self.uavs[jj].cur_pose, graph=self.graph)
        #     self.behavior_change[act] = bc
        #     self.action_values[act] = bc
            
        #     # self.action_values = dict(sorted(self.action_values.items(), key=lambda item: item[1], reverse=True))
        #     # self.actions = list(self.action_values.keys())[:min(param.MAX_UAV_ACTION, len(self.action_values))]
        #     # assert len(self.actions) <= param.MAX_UAV_ACTION
        #     # for _ in range(min(param.MAX_UAV_ACTION, len(self.action_values))):
        #     #     first_key = next(iter(self.action_values))
        #     #     self.action_values.pop(first_key)
        
        # uav_actions_left = [core.Action(target=poi.id, rtype=param.RobotType.Drone) for poi in new_graph.pois]
        # uav_actions_left = [action for action in uav_actions_left if self.history.get_action_outcome(action) == param.EventOutcome.CHANCE]
        # # self.uav_action_values.clear()
        # self.behavior_change.clear()
        # for action in uav_actions_left:
        #     bc_value = core.get_behavior_change(graph=new_graph, action=action, robot_edge=redge,
        #                                     d0=min_dist1, d1=min_dist2, goalID=self.goalID, atNode=self.robot.at_node,
        #                                     cur_heuristic=self.heuristic, n_samples=param.IV_SAMPLE_SIZE)
        #     self.behavior_change[action] = bc_value
    
    @property
    def is_goal_state(self):
        return all ([uav.last_node == self.restID for uav in self.uavs])

    @property
    def is_block_state(self):
        return self.noway2goal

    def transition(self, action):
        temp_state = self.copy()
        uav_needs_action = [uav.need_action for uav in temp_state.uavs]
        assert any(uav_needs_action) == True
        assert action in temp_state.actions
        assert action.target not in temp_state.assigned_pois
        uav_idx = uav_needs_action.index(True)
        start_pos = (temp_state.uavs[uav_idx].cur_pose[0], temp_state.uavs[uav_idx].cur_pose[1])
        if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
            ValueError("Start position is NaN") 
        action.update_pose(start_pos)
        action.update_robotID(uav_idx)
        distance, direction = temp_state.get_distance_direction(start_pos, action.target)            
        temp_state.uavs[uav_idx].retarget(action, distance, direction)
        if action.target != temp_state.restID:
            temp_state.assigned_pois.add(action.target)
            temp_state.actions.remove(action)
        if len(temp_state.actions) ==0:
            rest_action = core.Action(target=temp_state.restID, rtype=param.RobotType.Drone)
            temp_state.actions.append(rest_action)
        return advance_state(temp_state, action)

    def copy(self):
        new_state = DronesState(iscopy=True)
        new_state.vertices_map = self.vertices_map.copy()
        new_state.depth = self.depth
        new_state.graph = self.graph
        new_state.restID = self.restID
        new_state.going_back = False
        new_state.action_cost = 0.0
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.v_vertices = self.v_vertices.copy()
        new_state.heuristic = self.heuristic
        # copy the robot
        new_state.uavs = [uav.copy() for uav in self.uavs]
        new_state.actions = [core.Action(target=action.target, rtype=action.rtype, start_pose=action.start_pose) \
                                for action in self.actions]
        return new_state
            
    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

def advance_state(state, action):
    assert state.going_back == False
    # 1. if any robot needs action, determine its actions then return
    if any([uav.need_action for uav in state.uavs]):
        assert len(state.actions) > 0
        state.depth += 1
        return {state: (1.0, 0.0)}
    # 2. Find the robot that finishes its action first.
    uav_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    for uav in state.uavs:
        if uav.last_node != state.restID:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
    assert any([uav.need_action for uav in state.uavs]) == True
    if action in state.actions and action.target != state.restID:
        state.actions.remove(action)
    state.actions = [action for action in state.actions if action.target not in state.assigned_pois]    
    state.depth += 1
    return {state: (1.0, state.action_cost)}

def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    assert len(state.uavs) > 0
    for uav in state.uavs:
        if uav.last_node == state.restID:
            continue
        if uav.remaining_time > 0.0:
            time_remaining_uavs.append(uav.remaining_time)
    
    assert len(time_remaining_uavs) > 0
    min_time = min(time_remaining_uavs) if len(time_remaining_uavs) > 0 else 0.0
    remaining_times = [uav.remaining_time for uav in state.uavs]
    uav_index = remaining_times.index(min_time)
    return uav_index, min_time

def drone_rollout(node):
    return 0.0