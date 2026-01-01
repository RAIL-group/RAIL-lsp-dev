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
            while True:
                if self.uavs == []: # and len(self.ugvs)==1:
                    if discover:
                        init_actions_len = len(self.ugvs) - len([i for i, ugv in enumerate(self.ugvs) if ugv.last_node==self.goalIDs[i]])
                    else:
                        init_actions_len = len([ugv.need_action for ugv in self.ugvs if ugv.need_action==True and ugv.last_node!=self.goalIDs[self.ugvs.index(ugv)]])
                    discover = False
                    need_replan, discover, actions_list = self.baseline_move(actions_list, num_actions=init_actions_len)
                    
                else:
                    need_replan, actions_list = self.team_move(actions_list)
                all_robots_goal = all([ugv.last_node ==self.goalIDs[i] for i, ugv in enumerate(self.ugvs)])
                count += 1
                if need_replan or len(actions_list) == 0 or all_robots_goal:
                    self.reset_all_robots()
                    break
                elif discover:
                    self.reset_all_robots()
                    
                    
    def reset_all_robots(self):
        for ugv in self.ugvs:
            ugv.remaining_time = 0.0
            if ugv.last_node != self.goalIDs[self.ugvs.index(ugv)]:
                ugv.need_action = True
            else:
                ugv.need_action = False
        for drone in self.uavs:
            drone.remaining_time = 0.0
            if drone.last_node != self.goalIDs[0]:
                drone.need_action = True
            else:
                drone.need_action = False
        
            
    def team_move(self, actions_list):
        need_replan = True
        self.action_cost = self.update_joint_action(actions_list[:len(self.uavs)+len(self.ugvs)])    
        need_replan = self.transition_robots()
        if len(self.uavs) > 0:
            self.transition_drones()   
                
        actions_list = actions_list[len(self.uavs)+len(self.ugvs):]
        
        # if any([action.rtype == RobotType.Drone for action in actions_list]):
        #     need_replan = True
        # else:
        #     need_replan = False
        #     while actions_list != [] and not all([ugv.last_node ==self.goalIDs[i] for i, ugv in enumerate(self.ugvs)]):
        #         actions_list = self.update_onlyugv_action(actions_list)
        #         print("############################## Only move the ground robots ########################")                
        #         print(f"Remaining actions {len(actions_list)}")
        #         print(f"The action left are: {[print (action) for action in  actions_list]}")
        #         self.action_cost = min([ugv.remaining_time for i, ugv in enumerate(self.ugvs) if ugv.last_node != self.goalIDs[i]])
        #         assert self.action_cost > 0.0
        #         self.transition_robots()
        #         print(f"UGV positions: {[ugv.cur_pose for ugv in self.ugvs]} at node: {[ugv.last_node for ugv in self.ugvs]}")
        # Reset the robot and drones
        if need_replan:
            for ugv in self.ugvs:
                ugv.need_action = True
                ugv.remaining_time = 0.0
            for drone in self.uavs:
                drone.need_action = True 
                drone.remaining_time = 0.0
        return need_replan, actions_list
        
    def baseline_move(self, actions_list, num_actions=1):
        self.action_cost = self.update_joint_action(actions_list[:num_actions])    
        need_replan_robot, discover = self.transition_robots()
        
        return need_replan_robot, discover, actions_list[num_actions:]
    
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
        new_block_found = False
        for i, drone in enumerate(self.uavs):
            if drone.at_node and drone.last_node ==self.goalIDs[0]:
                continue
            drone.advance_time(self.action_cost)
            if drone.at_node: # sense the node
                drone.need_action = True 
                drone.remaining_time = 0.0
                vertex_id = drone.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    # sense the node
                    v[0].block_prob = float(v[0].block_status)
                    self.vertices_status[vertex_id] = v[0].block_status
                    new_block_found = True if v[0].block_status == 1 else False
                drone.unfinished_action = None
            else:
                drone.unfinished_action = drone.action
        return new_block_found
             

    def update_joint_action(self, joint_action):
        # in this function, all robots need action -
        if joint_action is None:
            return
        min_time = float('inf')
        if self.verbose:
            print(f"+++++++++++++++++++++++++ Updating {len(joint_action)} action(s) +++++++++++++++++")
            [print(action) for action in joint_action]
        for action in joint_action:
            robot_id = action.robotID
            if action.rtype == RobotType.Ground:
                ugv = self.ugvs[robot_id]
                if ugv.last_node == self.goalIDs[robot_id]:
                    continue
                if ugv.need_action == False:
                    print(f"Something is wrong: UGV {robot_id} is at node {ugv.last_node} node and goal is {self.goalIDs[robot_id]}")
                    print(f"UGV {robot_id} has need action? {ugv.need_action} and remaining time {ugv.remaining_time}")
                    for i, ugv in enumerate(self.ugvs):
                        if i != robot_id:
                            print(f"UGV {i}: last node {ugv.last_node}, goal {self.goalIDs[i]} need action? {ugv.need_action}, remaining time {ugv.remaining_time}")
                    raise ValueError("UGV is not ready to get a new action.")
                assert ugv.remaining_time == 0.0
                end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
                distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
                direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
                ugv.retarget(action, distance, direction)
                # print(f"Assigned action successfully for UGV {robot_id}!")
                min_time = min(distance, min_time)
            elif action.rtype == RobotType.Drone:
                drone = self.uavs[robot_id]
                assert drone.need_action == True
                assert drone.remaining_time == 0.0
                if drone.last_node == self.goalIDs[0]:
                    continue
                end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
                distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
                direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
                drone.retarget(action, distance, direction)
                min_time = min(distance/VEL_RATIO, min_time)
            else:
                raise ValueError("Unknown robot type in joint action")
        times_remaining = [ugv.remaining_time for i, ugv in enumerate(self.ugvs) if ugv.last_node != self.goalIDs[i]]
        assert times_remaining != []
        min_time = min(min_time, min(times_remaining))
        return min_time


    # def update_onlyugv_action(self, ugv_actions):
    #     assert ugv_actions is not None 
    #     if all ([ugv.remaining_time > 0.0 for i, ugv in enumerate(self.ugvs) if ugv.last_node != self.goalIDs[i]]):
    #         print("All UGVs are still executing their actions.")
    #         return ugv_actions
    #     while any([ugv.need_action for ugv in self.ugvs]):
    #         action = ugv_actions[0]
    #         if self.verbose:
    #             print("Updating only UGV actions...")
    #             print(f"Remaining UGV actions: {action}")
            
    #         ugv_actions = ugv_actions[1:]
    #         robot_id = action.robotID
    #         assert action.rtype == RobotType.Ground
    #         ugv = self.ugvs[robot_id]
    #         if ugv.need_action == False:
    #             print("Something is wrong")
    #             print(f"The robot ID is {robot_id} with need action {ugv.need_action} and remaining time {ugv.remaining_time}")
    #         assert ugv.need_action == True
    #         assert ugv.remaining_time <= APPROX_TIME
    #         end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
    #         distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
    #         direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])            
    #         ugv.retarget(action, distance, direction)
    #     return ugv_actions
        