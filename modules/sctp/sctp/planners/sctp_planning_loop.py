import numpy as np
import sctp
from sctp.param import VEL_RATIO

class SCTPPlanningLoop(object):
    def __init__(self, graph, reached_goal, goalID, robot, drones=None, verbose=True):
        self.robot = robot
        self.drones = drones
        self.graph = graph
        self.counter = 0
        self.verbose = verbose
        self.goalID = goalID
        # self.ordering = []
        self.goal_reached_fn = reached_goal
        self.action_cost = 0.0


    def __iter__(self):
        counter = 0
        vertices_status = {}
        while True:
            
            if self.drones:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node]),
                    "drones": ([[drone.cur_pose, drone.at_node, drone.last_node] for drone in self.drones]),
                    "observed_pois": (vertices_status)
                }
            else:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node]),
                    "drones": None,
                    "observed_pois": (vertices_status)
                }
            if self.goal_reached_fn():
                print("------ The ground robot reaches its goal ----- ")
                print(f"Drone position: ", [drone.cur_pose for drone in self.drones])
                print(f"Robot position: {self.robot.cur_pose}")
                break

            # # Compute the trajectory from robot's pose to the target node for each robot
            vertices_status.clear()
            self.robot.advance_time(self.action_cost)
            # sense the current point
            if self.robot.at_node:
                vertex_id = self.robot.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    vertices_status[vertex_id] = v[0].block_status
                    # v[0].block_prob = float(v[0].block_status)
            # # move the drones
            for i, drone in enumerate(self.drones):
                if drone.at_node and drone.last_node ==self.goalID:
                    continue
                drone.advance_time(self.action_cost)
                if drone.at_node: # sense the node
                    vertex_id = drone.last_node
                    v = [node for node in self.graph.pois if node.id == vertex_id]
                    if v:
                        vertices_status[vertex_id] = v[0].block_status
                        # v[0].block_prob = float(v[0].block_status)
            #Reset the robot and drones
            self.robot.need_action = True
            self.robot.remaining_time = 0.0
            for drone in self.drones:
                drone.need_action = True 
                drone.remaining_time = 0.0        

            counter += 1
            


    def update_joint_action(self, joint_action, action_cost):
        if joint_action is None:
            return
        # for the robot
        assert self.robot.remaining_time == 0.0
        assert self.robot.need_action == True
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[-1].target][0].coord
        distance = np.linalg.norm(self.robot.cur_pose - np.array(end_pos))
        if distance != 0.0:
            robot_direction = (np.array([end_pos[0], end_pos[1]]) - self.robot.cur_pose)/distance
        else:
            robot_direction = np.array([1.0, 1.0])
        self.robot.retarget(joint_action[-1], distance, robot_direction)
        min_time = distance

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
        self.action_cost = min_time
