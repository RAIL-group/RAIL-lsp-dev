import numpy as np
import mr_task


class MRTaskPlanningLoop(object):
    def __init__(self, robots, container_locations, object_locations, distance_fn, verbose=True):
        self.robots = robots
        self.container_locations = container_locations
        self.object_locations = object_locations
        self.objects_found = ()
        self.unexplored_containers = {coords: loc for coords, loc in container_locations.items()}
        self.joint_action = None
        self.distance_fn = distance_fn
        self.counter = 0
        self.verbose = verbose

    def __iter__(self):
        counter = 0
        while True:
            # make container nodes from unexplored containers
            container_nodes = [mr_task.core.Node(is_subgoal=True, name=loc, location=coords)
                               for coords, loc in self.unexplored_containers.items()]
            objects_found = self.objects_found
            self.known_space_nodes = [mr_task.core.Node(props=self.object_locations[location], name=location, location=coords)
                                      for coords, location in self.unexplored_containers.items()]

            yield {
                "robot_poses": [robot.pose for robot in self.robots],
                "container_nodes": container_nodes,
                "object_found": objects_found,
            }

            self.counter += 1
            distances = [self.distance_fn(robot, action.target_node)
                         for robot, action in zip(self.robots, self.joint_action)]
            first_revealed_action = self.joint_action[np.argmin(distances)]

            for robot, act in zip(self.robots, self.joint_action):
                robot.move(act.target_node.location, min(distances))

            # remove the object from the container
            location = self.unexplored_containers.pop(first_revealed_action.target_node.location)
            self.objects_found = tuple(self.object_locations[location])

            if self.verbose:
                for i, action in enumerate(self.joint_action):
                    print(f'R{i}->{(action.target_node.name, action.props)}', end=' ')
                print(f'\n{counter=}, {location=}, {self.objects_found=}')

            counter += 1

    def update_joint_action(self, joint_action):
        self.joint_action = joint_action
