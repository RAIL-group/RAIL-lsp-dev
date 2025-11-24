import numpy as np
import sctp
from sctp.param import VEL_RATIO, RobotType

class SCTPDecExe(object):
    def __init__(self, graph, goalIDs, robots, drones=[], verbose=False):
        self.robots = robots
        self.drones = drones
        self.graph = graph
        self.counter = 0
        self.verbose = verbose
        self.goalIDs = goalIDs
        # self.goal_reached_fns = reached_goals
        self.immed_next_cost = 0.0
        self.drones_actions = []
        self.drones_costs = []
        self.robots_actions = []
        self.robots_costs = []
        self.max_counter = 100
        self.counter = 0
        self.success = False
        self.vertices_status = {}

    def __iter__(self):
        while True:
            if len(self.drones) > 0:
                yield {
                    "robots": ([[r.cur_pose, r.at_node, r.edge, r.last_node, r.pl_vertex] for r in self.robots]),
                    "drones": ([[d.cur_pose, d.at_node, d.last_node, d.unfinished_action] for d in self.drones]),
                    "observed_pois": (self.vertices_status)
                }
            else:
                yield {
                    "robots": ([[r.cur_pose, r.at_node, r.edge, r.last_node, r.pl_vertex] for r in self.robots]),
                    "drones": None,
                    "observed_pois": (self.vertices_status)
                }
            if self.all_reached_goals():
                print("----------------- all ground robot reach their goals ---------------- ")
                self.success = True
                if self.verbose:
                    print(f"Robot positions: {[robot.cur_pose for robot in self.robots]}")
                    if len(self.drones) > 0:
                        print(f"Drone positions: ", [drone.cur_pose for drone in self.drones])
                break
            if self.counter > self.max_counter:
                self.success = False
                print("################# Robot failed to find path to goal ##########################")
                break
            self.vertices_status.clear()
            self.counter += 1
            count = 0
            while True:
                # if self.drones == []:
                #     need_replan, actions_list = self.baseline_move(actions_list)
                # else:
                robots_actions = self.robots_actions
                drones_actions = self.drones_actions
                need_replan, _ = self.team_move(robots_actions, drones_actions)
                count += 1
                # if need_replan or len(actions_list) == 0:
                #     break
                break
            
    def all_reached_goals(self):
        return all([self.goalIDs[i] == self.robots[i].last_node for i in range(len(self.robots))])
    
    def team_move(self, robots_actions, drones_actions):
        need_replan = True
        robots_action = [action[0] for action in robots_actions if action is not None]
        
        assert len(robots_action) == len(self.robots), "The number of robot actions must be equal to the number of robots"
        self.immed_next_cost = self.min_action_cost(robots_action, drones_actions[:len(self.drones)])    
        self.transition_robot()
        if len(self.drones) > 0:
            self.transition_drones()
        # reset the robot and drones:
        for robot in self.robots:
            robot.need_action = True
            robot.remaining_time = 0.0 
        for drone in self.drones:
            drone.need_action = True 
            drone.remaining_time = 0.0
        return need_replan, robots_action[len(self.robots):]
        
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
    
    def save_joint_actions(self, robots_actions, robot_costs, drones_actions=[], drones_costs=[]):
        self.robots_actions = robots_actions
        self.robots_costs = robot_costs
        self.drones_actions = drones_actions
        self.drones_costs = drones_costs

    def update_action(self, action, rd_ID = None):        
        if action.rtype == RobotType.Ground:
            assert self.robot.remaining_time == 0.0
            assert self.robot.need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
            distance = np.linalg.norm(np.array(self.robot.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0],end_pos[1]])-self.robot.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            self.robot.retarget(action, distance, direction)
        else:
            assert rd_ID is not None, "Drone ID must be provided for drone actions"
            drone = self.drones[rd_ID]
            assert drone.remaining_time == 0.0
            assert drone.need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
            distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            drone.retarget(action, distance, direction)
        uav_remaining_time = [uav.remaining_time for uav in self.drones if uav.remaining_time > 0.0]
        if len(uav_remaining_time) == 0:
            min_time = self.robot.remaining_time
        else:
            min_time = min(uav_remaining_time) if uav_remaining_time else self.robot.remaining_time
            min_time = min(min_time, self.robot.remaining_time)
        return min_time
            
    def transition_robot(self):
        new_block_found = False
        for i, robot in enumerate(self.robots):
            if robot.at_node and robot.last_node ==self.goalIDs[i]:
                if self.verbose:
                    print(f"Robot {i}: is at goal: {robot.last_node}")
                continue
            robot.advance_time(self.immed_next_cost)
            if self.verbose:
                print(f"Robot {i}: {robot.action}")
            # sense the current point
            if robot.at_node:
                robot.need_action = True
                robot.remaining_time = 0.0
                vertex_id = robot.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    self.vertices_status[vertex_id] = v[0].block_status
                    v[0].block_prob = float(v[0].block_status)
                    new_block_found = True if v[0].block_status == 1 else False
                        
        return new_block_found

    def transition_drones(self):
        new_block_found = False
        for i, drone in enumerate(self.drones):
            if drone.at_node and drone.last_node ==self.goalIDs[0]:
                if self.verbose:
                    print(f"Drone {i}: is at goal: Node_{drone.last_node}")
                continue
            drone.advance_time(self.immed_next_cost)
            if self.verbose:
                print(f"Drone {i}: {drone.action}")

            if drone.at_node: # sense the node
                drone.need_action = True 
                drone.remaining_time = 0.0
                vertex_id = drone.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    v[0].block_prob = float(v[0].block_status)
                    self.vertices_status[vertex_id] = v[0].block_status
                    new_block_found = True if v[0].block_status == 1 else False
                drone.unfinished_action = None
            else:
                drone.unfinished_action = drone.action
        return new_block_found
             

    def min_action_cost(self, robots_action, drones_actions):
        min_time = float('inf')
        if robots_action is None:
            return 0.0
        # for the robots
        for ii, robot in enumerate(self.robots):
            if robot.at_node and robot.last_node == self.goalIDs[ii]:
                continue
            assert robot.need_action == True
            assert robot.remaining_time == 0.0
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == robots_action[ii].target][0].coord
            distance = np.linalg.norm(np.array(robot.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - robot.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            robot.retarget(robots_action[ii], distance, direction)
            min_time = distance if (distance < min_time) else min_time
            
        # for drones
        for ii, drone in enumerate(self.drones):
            if drone.at_node and drone.last_node == self.goalIDs[0]:
                continue
            assert drone.remaining_time == 0.0
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == drones_actions[ii].target][0].coord
            distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            drone.retarget(drones_actions[ii], distance, direction)
            if distance > 0.0 and (distance/VEL_RATIO) < min_time:
                min_time = distance/VEL_RATIO
            # else:
            #     if drone.remaining_time < min_time and drone.remaining_time > 0.0:
            #         min_time = drone.remaining_time
        return min_time
