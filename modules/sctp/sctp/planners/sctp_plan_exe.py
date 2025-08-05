import numpy as np
import sctp
from sctp.param import VEL_RATIO

class SCTPPlanExecution(object):
    def __init__(self, graph, reached_goal, goalID, robot, drones=[], verbose=True):
        self.robot = robot
        self.drones = drones
        self.graph = graph
        self.counter = 0
        self.verbose = verbose
        self.goalID = goalID
        self.goal_reached_fn = reached_goal
        self.action_cost = 0.0
        self.joint_actions = []
        self.costs = []
        self.max_counter = 30
        self.counter = 0
        self.success = False
        self.vertices_status = {}


    def __iter__(self):
        while True:
            
            if len(self.drones) > 0:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node, self.robot.pl_vertex]),
                    "drones": ([[drone.cur_pose, drone.at_node, drone.last_node] for drone in self.drones]),
                    "observed_pois": (self.vertices_status)
                }
            else:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node,self.robot.pl_vertex]),
                    "drones": None,
                    "observed_pois": (self.vertices_status)
                }
            if self.goal_reached_fn():
                print("----------------- The ground robot reaches its goal ---------------- ")
                self.success = True
                if self.verbose:
                    print(f"Robot position: {self.robot.cur_pose}")
                    if len(self.drones) > 0:
                        print(f"Drone position: ", [drone.cur_pose for drone in self.drones])
                break
            if self.counter > self.max_counter:
                self.success = False
                print("################# Robot failed to find path to goal ##########################")
                break
            self.vertices_status.clear()
            actions_list = [action for action in self.joint_actions]
            self.counter += 1
            while True:
                if self.drones == []:
                    need_replan, actions_list = self.baseline_move(actions_list)
                else:
                    need_replan = self.team_move(actions_list[:len(self.drones)+1])
                if need_replan or len(actions_list) == 0:
                    break
                # if self.robot.need_action and actions_list[0].rtype == sctp.param.RobotType.Ground:
                #     min_time = min(min_time, self.update_action(actions_list[0]))
                #     actions_list = actions_list[1:]
                #     if self.is_anyrobot_needaction():
                #         continue
                # if len(actions_list) >0 and actions_list[0].rtype == sctp.param.RobotType.Drone and \
                #         any([drone.need_action for drone in self.drones if drone.last_node != self.goalID]):
                #     # assert 1==0
                #     for i, drone in enumerate(self.drones):
                #         if np.linalg.norm(drone.cur_pose-actions_list[0].start_pose) < 0.1 and drone.need_action == True:
                #         # if drone.cur_pose[0] == actions_list[0].start_pose[0] and drone.cur_pose[1]==actions_list[0].start_pose[1]:
                #             min_time = min(min_time, self.update_action(actions_list[0], droneID=i))
                #             actions_list = actions_list[1:]
                #             break
                #     if self.is_anyrobot_needaction():
                #         continue
                # assert self.is_anyrobot_needaction() == False
                # self.action_cost = min_time
                # for drone in self.drones:
                #     if 0.0 < drone.remaining_time < self.action_cost:
                #         self.action_cost = drone.remaining_time
                # if 0.0 < self.robot.remaining_time < self.action_cost:
                #     self.action_cost = self.robot.remaining_time
                
                # need_replan_robot = self.transition_robot()
                # need_replan_drones = False
                # if len(self.drones) > 0:
                #     need_replan_drones = self.transition_drones()
                # min_time = 10e6
                # if need_replan_robot or need_replan_drones:
                #     break
                # number_action += 1
                
            
            # for i, action in enumerate(self.joint_actions):
            #     need_replan = False
            #     self.update_joint_action([action], 0.0)    
            #     # Compute the trajectory from robot's pose to the target node for each robot
            #     self.robot.advance_time(self.action_cost)
            #     # sense the current point
            #     if self.robot.at_node:
            #         vertex_id = self.robot.last_node
            #         v = [node for node in self.graph.pois if node.id == vertex_id]
            #         if v:
            #             self.vertices_status[vertex_id] = v[0].block_status
            #             v[0].block_prob = float(v[0].block_status)
            #             if v[0].block_status == 1:
            #                 print(f"Robot senses node {vertex_id} is blocked")
            #                 need_replan = True
            #     # move the drones
            #     for i, drone in enumerate(self.drones):
            #         if drone.at_node and drone.last_node ==self.goalID:
            #             continue
            #         drone.advance_time(self.action_cost)
            #         if drone.at_node: # sense the node
            #             vertex_id = drone.last_node
            #             v = [node for node in self.graph.pois if node.id == vertex_id]
            #             if v:
            #                 self.vertices_status[vertex_id] = v[0].block_status
            #                 v[0].block_prob = v[0].block_status
            #     #Reset the robot and drones
            #     self.robot.need_action = True
            #     self.robot.remaining_time = 0.0
            #     for drone in self.drones:
            #         drone.need_action = True 
            #         drone.remaining_time = 0.0
            #     if need_replan or i >= 2:
            #         break
            
    def team_move(self, actions_list):
        self.action_cost = self.update_joint_action(actions_list)    
        self.transition_robot()
        self.transition_drones()
        # Reset the robot and drones
        self.robot.need_action = True
        self.robot.remaining_time = 0.0
        for drone in self.drones:
            drone.need_action = True 
            drone.remaining_time = 0.0
        return True
        
    def baseline_move(self, actions_list):
        self.action_cost = self.update_action(actions_list[0])
        need_replan_robot = self.transition_robot()
        return need_replan_robot, actions_list[1:]
    
    def is_anyrobot_needaction(self):
        if self.robot.need_action:
            return True 
        if len(self.drones) > 0 and any([drone.need_action for drone in self.drones]):
            return True
        return False
    
    def save_joint_actions(self, joint_actions, costs):
        self.joint_actions = joint_actions
        self.costs = costs

    def update_action(self, action, droneID = None):        
        # if droneID is None:
        # for the robot
        assert self.robot.remaining_time == 0.0
        assert self.robot.need_action == True
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
        distance = np.linalg.norm(self.robot.cur_pose - np.array(end_pos))
        if distance != 0.0:
            robot_direction = (np.array([end_pos[0], end_pos[1]]) - self.robot.cur_pose)/distance
        else:
            robot_direction = np.array([1.0, 1.0])
        self.robot.retarget(action, distance, robot_direction)
        return distance
        # else:
        #     assert self.drones[droneID].remaining_time == 0.0
        #     assert self.drones[droneID].need_action == True
        #     end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
        #     distance = np.linalg.norm(self.drones[droneID].cur_pose - np.array(end_pos))            
        #     if distance != 0.0:
        #         direction = (np.array([end_pos[0], end_pos[1]]) - self.drones[droneID].cur_pose)/distance
        #     else:
        #         direction = np.array([1.0, 1.0])
        #     self.drones[droneID].retarget(action, distance, direction)
        #     min_time = 0.5*distance
        # return min_time
            
    def transition_robot(self):
        self.robot.advance_time(self.action_cost)
        # sense the current point
        if self.robot.at_node:
            self.robot.need_action = True
            self.robot.remaining_time = 0.0
            vertex_id = self.robot.last_node
            v = [node for node in self.graph.pois if node.id == vertex_id]
            if v:
                self.vertices_status[vertex_id] = v[0].block_status
                v[0].block_prob = float(v[0].block_status)
                if v[0].block_status == 1:
                    return True
        return False

    def transition_drones(self):
        new_block_found = False
        for i, drone in enumerate(self.drones):
            if drone.at_node and drone.last_node ==self.goalID:
                continue
            drone.advance_time(self.action_cost)
            if drone.at_node: # sense the node
                drone.need_action = True 
                drone.remaining_time = 0.0
                vertex_id = drone.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    self.vertices_status[vertex_id] = v[0].block_status
                    v[0].block_prob = float(v[0].block_status)
                    if v[0].block_status == 1:
                        new_block_found = True
        return new_block_found
             

    def update_joint_action(self, joint_action):
        # in this function, all robots need action -
        if joint_action is None:
            return
        # for the robot
        if self.robot.need_action:
            assert self.robot.remaining_time == 0.0
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[-1].target][0].coord
            distance = np.linalg.norm(self.robot.cur_pose - np.array(end_pos))
            if distance != 0.0:
                robot_direction = (np.array([end_pos[0], end_pos[1]]) - self.robot.cur_pose)/distance
            else:
                robot_direction = np.array([1.0, 1.0])
            self.robot.retarget(joint_action[-1], distance, robot_direction)
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
            distance = np.linalg.norm(drone.cur_pose - np.array(end_pos))            
            if distance != 0.0:
                direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance
            else:
                direction = np.array([1.0, 1.0])
            drone.retarget(joint_action[i], distance, direction)
            if distance > 0.0 and 0.5*distance < min_time:
                min_time = 0.5*distance
        return min_time
        # self.action_cost = min_time
