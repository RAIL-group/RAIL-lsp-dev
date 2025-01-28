import numpy as np


def get_inter_distances_nodes(nodes, robot_nodes):
    distances = {
        (node1, node2): np.linalg.norm(np.array(node1.location) - np.array(node2.location))
        for node1 in nodes
        for node2 in nodes
    }
    distances.update({
        (robot_node.start, node): np.linalg.norm(np.array(robot_node.start.location) - np.array(node.location))
        for robot_node in robot_nodes
        for node in nodes
    })
    return distances
