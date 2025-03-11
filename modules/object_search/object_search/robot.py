import copy


class Robot:
    def __init__(self, pose):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]

    def move(self, pose):
        self.pose = pose
        self.all_poses.append(copy.copy(self.pose))
