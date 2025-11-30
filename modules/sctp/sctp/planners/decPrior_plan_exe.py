import numpy as np
import sctp
from sctp.param import VEL_RATIO, RobotType

class DecPriorPlanExe(object):
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
            # print(f"The drone unfinished action is {self.drones[0].unfinished_action}")
            if len(self.drones) > 0:
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
                        print(f"Drone position: ", [drone.cur_pose for drone in self.drones])
                break
            if self.counter > self.max_counter:
                self.success = False
                print("################# Robot failed to find path to goal ##########################")
                break
            self.vertices_status.clear()
            actions_list = [action for action in self.joint_actions]
            self.counter += 1
            # first_exec = True
            count = 0
            while True:
                # if self.uavs == []:
                #     need_replan, actions_list = self.baseline_move(actions_list)
                # else:
                    # print(f"Multi-movements - step {count}, remaining actions {len(actions_list)}")
                need_replan, actions_list = self.team_move(actions_list)
                count += 1
                if need_replan or len(actions_list) == 0:
                    break
                
            
    def team_move(self, actions_list):
        need_replan = True
        # if actions_list[0].rtype == RobotType.Drone:
        self.action_cost = self.update_joint_action(actions_list[:len(self.uavs)+len(self.ugvs)])    
        self.transition_robots()
        self.transition_drones()
        actions_list = actions_list[len(self.uavs)+len(self.ugvs):]
        # Reset the robot and drones
        if need_replan:
            for ugv in self.ugvs:
                ugv.need_action = True
                ugv.remaining_time = 0.0
            for drone in self.uavs:
                drone.need_action = True 
                drone.remaining_time = 0.0
        return need_replan, actions_list
        
    # def baseline_move(self, actions_list):
    #     self.action_cost = self.update_action(actions_list[0])
    #     need_replan_robot = self.transition_robot()
    #     return need_replan_robot, actions_list[1:]
    
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
        for i, ugv in enumerate(self.ugvs):
            if ugv.at_node and ugv.last_node ==self.goalIDs[i]:
                continue
            ugv.advance_time(self.action_cost)
            # sense the current point
            if ugv.at_node:
                ugv.need_action = True
                ugv.remaining_time = 0.0
                vertex_id = ugv.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    self.vertices_status[vertex_id] = v[0].block_status
                    v[0].block_prob = float(v[0].block_status)
                    if v[0].block_status == 1:
                        return True
        return False

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
        # for the robot
        for i, ugv in enumerate(self.ugvs):
            if ugv.need_action:
                assert ugv.remaining_time == 0.0
                // need to work on this
                end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[-1].target][0].coord
                distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
                direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
                
                // check check 
                ugv.retarget(joint_action[-1], distance, direction)
                min_time = distance
            else:
                min_time = self.robot.remaining_time
            
        # for drones
        for i, drone in enumerate(self.drones):
            if drone.last_node == self.goalID:
                continue
            assert drone.remaining_time == 0.0
            assert drone.need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[i].target][0].coord
            distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            drone.retarget(joint_action[i], distance, direction)
            if distance > 0.0 and (distance/VEL_RATIO) < min_time:
                min_time = distance/VEL_RATIO
        return min_time
