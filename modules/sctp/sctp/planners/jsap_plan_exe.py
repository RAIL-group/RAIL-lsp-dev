import numpy as np
import sctp
from sctp.param import VEL_RATIO, RobotType, APPROX_TIME

class JSAPPlanExe(object):
    def __init__(self, graph, reached_goal, goalIDs, ugvs, uavs=[], verbose=True):
        self.ugvs = ugvs
        self.uavs = uavs
        self.graph = graph
        self.counter = 0
        self.verbose = verbose
        self.goalIDs = goalIDs
        self.goal_reached_fn = reached_goal
        self.action_cost = 0.0
        self.joint_actions = []
        self.costs = []
        self.max_counter = 100
        self.counter = 0
        self.success = False
        self.vertices_status = {}

    def __iter__(self):
        while True:
            if len(self.uavs) > 0:
                yield {
                    "ugvs": ([[ugv.cur_pose, ugv.at_node, ugv.edge, ugv.last_node, ugv.pl_vertex] for ugv in self.ugvs]),
                    "uavs": ([[uav.cur_pose, uav.at_node, uav.last_node, uav.unfinished_action] for uav in self.uavs]),
                    "observed_pois": (self.vertices_status)
                }
            else:
                yield {
                    "ugvs": ([[ugv.cur_pose, ugv.at_node, ugv.edge, ugv.last_node, ugv.pl_vertex] for ugv in self.ugvs]),
                    "uavs": None,
                    "observed_pois": (self.vertices_status)
                }
            if self.goal_reached_fn():
                print("----------------- The ground robot reaches its goal ---------------- ")
                self.success = True
                if self.verbose:
                    print(f"Robot's positions: {[ugv.cur_pose for ugv in self.ugvs]}")
                    if len(self.uavs) > 0:
                        print(f"Drone position: ", [drone.cur_pose for drone in self.uavs])
                break
            if self.counter > self.max_counter:
                self.success = False
                print("################# Robot failed to find path to goal ##########################")
                break
            self.vertices_status.clear()
            actions_list = [action for action in self.joint_actions]
            self.counter += 1
            count = 0
            discover = False
            init_move = True
            while True:
                uavs_reset = []    
                if self.uavs == []:
                    if init_move:
                        init_move = False
                        init_actions_len = len(self.ugvs) + len(self.uavs)
                    else:
                        if discover:
                            init_actions_len = len(self.ugvs) - len([i for i, ugv in enumerate(self.ugvs) if ugv.last_node==self.goalIDs[i]])
                        else:
                            init_actions_len = len([ugv.need_action for ugv in self.ugvs if ugv.need_action==True and ugv.last_node!=self.goalIDs[self.ugvs.index(ugv)]])
                        if actions_list[0].robotID in [self.ugvs.index(ugv) for ugv in self.ugvs if ugv.last_node==self.goalIDs[self.ugvs.index(ugv)]]:
                            init_actions_len += 1
                    discover = False
                    need_replan, discover, actions_list = self.baseline_move(actions_list, num_actions=init_actions_len)
                    if self.verbose:
                        print(f"Discover new information? {discover}")
                else:
                    if init_move:
                        init_move = False
                        num_act = len(self.ugvs) + len(self.uavs)
                    else:
                        num_act = self.get_num_actions_needed(actions_list, uavs_reset, discover)
                    discover = False
                    need_replan, discover, actions_list = self.team_move(actions_list, num_actions=num_act)
                    if self.verbose:
                        print(f"Discover new information? {discover}")
                all_robots_goal = all([ugv.last_node ==self.goalIDs[i] for i, ugv in enumerate(self.ugvs)])
                count += 1
                if need_replan or len(actions_list) == 0 or all_robots_goal:
                    self.reset_all_robots()
                    break
                elif discover:
                    self.reset_ugvs()
                    
    def get_num_actions_needed(self, actions_list, uavs_reset, discover=False):
        if discover:
            al = len(self.ugvs)-len([i for i, ugv in enumerate(self.ugvs) if ugv.last_node==self.goalIDs[i]])
            al1 = set([ugv.last_node for ugv in self.ugvs if ugv.need_action])
            al2 = [uav.action.target for uav in self.uavs if uav.need_action==False]
            count_same_target = 0
            for target in al2:
                if target in al1:
                    uav_id = [i for i, uav in enumerate(self.uavs) if uav.action.target==target and uav.need_action==False]
                    assert len(uav_id) == 1
                    uavs_reset.append(uav_id[0])
                    count_same_target += 1
            al += count_same_target
            if uavs_reset != []:
                self.reset_some_uavs(uavs_reset)
        else:
            alen1 = len([ugv.need_action for ugv in self.ugvs \
                                    if ugv.need_action==True and ugv.last_node!=self.goalIDs[self.ugvs.index(ugv)]])
            alen2 = len([uav.need_action for uav in self.uavs \
                                    if uav.need_action==True and uav.last_node!=self.goalIDs[0]])
            al = alen1 + alen2
        if actions_list[0].robotID in [i for i, ugv in enumerate(self.ugvs) if ugv.last_node==self.goalIDs[i]]:
            al += 1
        return al
                      
                    
    def reset_all_robots(self):
        self.reset_ugvs()
        for uav in self.uavs:
            uav.remaining_time = 0.0
            if uav.last_node != self.goalIDs[0]:
                uav.need_action = True
            else:
                uav.need_action = False
    
    def reset_ugvs(self):
        for ugv in self.ugvs:
            ugv.remaining_time = 0.0
            if ugv.last_node != self.goalIDs[self.ugvs.index(ugv)]:
                ugv.need_action = True
            else:
                ugv.need_action = False
    
    def reset_some_uavs(self, uav_ids):
        if self.verbose:
            print(f"reseting the following UAVs {uav_ids}")
        for i, uav in enumerate(self.uavs):
            if i in uav_ids:
                uav.remaining_time = 0.0
                uav.need_action = True
                
            
    def team_move(self, actions_list, num_actions=1):
        need_replan2 = False
        # self.action_cost = self.update_joint_action(actions_list[:num_actions])
        self.action_cost, num_actions = self.update_joint_action(actions_list)
        if len(self.uavs) > 0:
            need_replan2, discover2 = self.transition_drones()
        need_replan1, discover1 = self.transition_robots()        
        # return need_replan1 or need_replan2, discover1 or discover2, actions_list[num_actions:]
        return discover1 or discover2, discover1 or discover2, actions_list[num_actions:]
        
    def baseline_move(self, actions_list, num_actions=1):
        # self.action_cost = self.update_joint_action(actions_list[:num_actions])
        self.action_cost, num_actions = self.update_joint_action(actions_list)    
        need_replan_robot, discover = self.transition_robots()
        
        # return need_replan_robot, discover, actions_list[num_actions:]
        return discover, discover, actions_list[num_actions:]
    
    def is_anyrobot_needaction(self):
        if any([ugv.need_action for ugv in self.ugvs]):
            return True 
        if len(self.drones) > 0 and any([drone.need_action for drone in self.drones]):
            return True
        return False
    
    def save_joint_actions(self, joint_actions, costs):
        self.joint_actions = joint_actions
        self.costs = costs

    def update_action(self, action, rd_ID):        
        if action.rtype == RobotType.Ground:
            assert self.ugvs[rd_ID].remaining_time == 0.0
            assert self.ugvs[rd_ID].need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
            distance = np.linalg.norm(np.array(self.ugvs[rd_ID].cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0],end_pos[1]])-self.ugvs[rd_ID].cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            self.ugvs[rd_ID].retarget(action, distance, direction)
        else:
            assert rd_ID is not None, "Drone ID must be provided for drone actions"
            drone = self.uavs[rd_ID]
            assert drone.remaining_time == 0.0
            assert drone.need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
            distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            drone.retarget(action, distance, direction)
        uav_remaining_times = [uav.remaining_time for uav in self.uavs if uav.remaining_time > 0.0]
        ugv_remaining_times = [ugv.remaining_time for ugv in self.ugvs if ugv.remaining_time > 0.0]
        
        # ugv_min_time = min([ugv.remaining_time for ugv in self.ugvs if ugv.last_node !=self.goalIDs[rd_ID]])
        if len(uav_remaining_times) == 0:
            min_time = min(ugv_remaining_times) if ugv_remaining_times else 0.0
        else:
            min_time = min(min(uav_remaining_times), min(ugv_remaining_times))
        return min_time
            
    def transition_robots(self):
        discover = False
        replan = False
        for i, ugv in enumerate(self.ugvs):
            if ugv.at_node and ugv.last_node ==self.goalIDs[i]:
                continue
            ugv.advance_time(self.action_cost)
            if self.verbose:
                print(f"UGV {i} advanced time by {self.action_cost}, remaining time: {ugv.remaining_time}")
            # sense the current point
            if ugv.at_node:
                ugv.need_action = True
                ugv.remaining_time = 0.0
                vertex_id = ugv.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v != [] and 0.0 < v[0].block_prob <1.0:
                    discover = True
                    self.vertices_status[vertex_id] = v[0].block_status
                    v[0].block_prob = float(v[0].block_status)
                    if v[0].block_status == 1:
                        replan = True
        return replan, discover

    def transition_drones(self):
        replan = False
        for i, drone in enumerate(self.uavs):
            if drone.at_node and drone.last_node ==self.goalIDs[0]:
                continue
            drone.advance_time(self.action_cost)
            if self.verbose:
                print(f"UAV {i} advanced time by {self.action_cost}, remaining time: {drone.remaining_time}")
            if drone.at_node: # sense the node
                drone.need_action = True 
                drone.remaining_time = 0.0
                vertex_id = drone.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v != [] and 0.0 < v[0].block_prob < 1.0:
                    v[0].block_prob = float(v[0].block_status)
                    self.vertices_status[vertex_id] = v[0].block_status
                    replan = True
                drone.unfinished_action = None
            else:
                drone.unfinished_action = drone.action
        return replan, replan
             

    def update_joint_action(self, joint_action):
        if joint_action is None:
            return
        number_action = 0
        
        for ii in range(len(self.ugvs)+len(self.uavs)):
            if ii >= len(joint_action):
                break
            action = joint_action[ii]
            robot_id = action.robotID
            if action.rtype == RobotType.Ground:
                ugv = self.ugvs[robot_id]
                if ugv.last_node == self.goalIDs[robot_id]:
                    number_action += 1
                    continue
                if ugv.need_action == False:
                    break
                assert ugv.remaining_time == 0.0
                end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
                distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
                direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
                ugv.retarget(action, distance, direction)
                number_action += 1
                if self.verbose:
                    print(f"Assigned action for UGV {robot_id} with remaining time: {ugv.remaining_time}!")
            elif action.rtype == RobotType.Drone:
                drone = self.uavs[robot_id]
                if drone.last_node == self.goalIDs[0]:
                    number_action += 1
                    continue
                if drone.need_action == False:
                    break
                assert drone.remaining_time == 0.0
                end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
                distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
                direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
                drone.retarget(action, distance, direction)
                number_action += 1
                if self.verbose:
                    print(f"Assigned action for UAV {robot_id} with remaining time: {drone.remaining_time}!")
            else:
                raise ValueError("Unknown robot type in joint action")
        
        if self.verbose:
            print(f"+++ Total: updating {number_action} action(s) +++++++++++++++++")
            [print(action) for action in joint_action[:number_action]]
            print("----------------------------------------------------------------------")
        
        # if self.verbose:
        #     print(f"+++++++++++++++++++++++++ Updating {len(joint_action)} action(s) +++++++++++++++++")
        #     [print(action) for action in joint_action]
        # for action in joint_action:
        #     robot_id = action.robotID
        #     if action.rtype == RobotType.Ground:
        #         ugv = self.ugvs[robot_id]
        #         if ugv.last_node == self.goalIDs[robot_id]:
        #             continue
        #         if ugv.need_action == False:
        #             print(f"Something is wrong: UGV {robot_id} is at node {ugv.last_node} node and goal is {self.goalIDs[robot_id]}")
        #             print(f"UGV {robot_id} has need action? {ugv.need_action} and remaining time {ugv.remaining_time}")
        #             for i, ugv in enumerate(self.ugvs):
        #                 if i != robot_id:
        #                     print(f"UGV {i}: last node {ugv.last_node}, goal {self.goalIDs[i]} need action? {ugv.need_action}, remaining time {ugv.remaining_time}")
        #             raise ValueError("UGV is not ready to get a new action.")
        #         assert ugv.remaining_time == 0.0
        #         end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
        #         distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
        #         direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
        #         ugv.retarget(action, distance, direction)
        #         if self.verbose:
        #             print(f"Assigned action for UGV {robot_id} with remaining time: {ugv.remaining_time}!")
        #     elif action.rtype == RobotType.Drone:
        #         drone = self.uavs[robot_id]
        #         if drone.last_node == self.goalIDs[0]:
        #             continue
        #         if drone.need_action == False:
        #             print(f"Something is wrong: UAV {robot_id} is at node {drone.last_node} node and goal is {self.goalIDs[0]}")
        #             print(f"UAV {robot_id} has need action? {drone.need_action} and remaining time {drone.remaining_time}")
        #             for i, uav in enumerate(self.uavs):
        #                 if i != robot_id:
        #                     print(f"UAV {i}: last node {uav.last_node}, goal {self.goalIDs[0]} need action? {uav.need_action}, remaining time {uav.remaining_time}")
        #             raise ValueError("UAV is not ready to get a new action.")
        #         assert drone.remaining_time == 0.0
        #         end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
        #         distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
        #         direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
        #         drone.retarget(action, distance, direction)
        #         if self.verbose:
        #             print(f"Assigned action for UAV {robot_id} with remaining time: {drone.remaining_time}!")
        #     else:
        #         raise ValueError("Unknown robot type in joint action")
        
        
        
        
        
        times_ugvs_remaining = [ugv.remaining_time for i, ugv in enumerate(self.ugvs) if ugv.last_node != self.goalIDs[i]]
        assert times_ugvs_remaining != []
        min_time1 = min(times_ugvs_remaining)
        if len(self.uavs) > 0:
            times_uavs_remaining = [uav.remaining_time for uav in self.uavs if uav.last_node != self.goalIDs[0]]
            if times_uavs_remaining != []:
                min_time1 = min(min_time1, min(times_uavs_remaining))
        return min_time1, number_action