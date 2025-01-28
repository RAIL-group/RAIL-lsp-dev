import copy
import random
import numpy as np


class Robot:
    def __init__(self, pose, map_data=None):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(pose)]
        self.net_motion = 0

    def move(self, coords, distance):
        start = np.array((self.pose.x, self.pose.y))
        end = np.array(coords)
        self.pose.x, self.pose.y = self._get_coordinates_after_distance(start, end, distance)
        self.net_motion += distance
        self.all_poses.append(copy.copy(self.pose))

    def _get_coordinates_after_distance(self, start, end, distance):
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        return start + direction * distance
