import yaml
from procthor.scenegraph import SceneGraph
import math
from procthor.utils import get_dc_comps
# from scipy.spatial.transform import Rotation as R
from . import ros_utils
import numpy as np


def get_scene_graph_from_yaml(yaml_file):

    with open(yaml_file, "r") as f:
        scene_data = yaml.safe_load(f)
    print(scene_data)
    """Create a scene graph from parsed YAML scene data."""
    graph = SceneGraph()

    # Add apartment node
    apartment_idx = graph.add_node({
        'id': 'Apartment|0',
        'name': 'apartment',
        'pos': (0, 0, 0),
        'type': [1, 0, 0, 0]  # Apartment
    })

    # Add rooms
    room_containers_info = {}
    for room_name, room_data in scene_data.items():
        room_id = f"{room_name}|0"
        room_position = (room_data['x'], room_data['y'], room_data['yaw'])
        room_idx = graph.add_node({
            'id': room_id,
            'name': room_name.lower(),
            'pos': room_position,
            'type': [0, 1, 0, 0]  # Room
        })
        graph.add_edge(apartment_idx, room_idx)
        room_containers_info[room_idx] = room_data.get('containers', {})

    for room_idx, containers in room_containers_info.items():
        # Add containers in the room
        room_id = graph.nodes[room_idx]['id']
        for container_name, pos in containers.items():
            container_id = f"{room_id}|{container_name}"
            container_position = (pos['x'], pos['y'], pos['yaw'])

            container_idx = graph.add_node({
                'id': container_id,
                'name': container_name.lower(),
                'pos': container_position,
                'type': [0, 0, 1, 0]  # Container
            })
            graph.add_edge(room_idx, container_idx)

    ensure_connectivity(graph)
    return graph


def ensure_connectivity(graph):
    """Ensure the graph is connected by adding edges between rooms using Euclidean distance."""
    required_edges = get_edges_for_connected_graph({
        'nodes': graph.nodes,
        'edge_index': graph.edges,
        'room_node_idx': graph.room_indices,
    }, pos='pos')
    graph.edges.extend(required_edges)


def get_edges_for_connected_graph(graph, pos='pos'):
    """ This function finds edges that needs to exist to have a connected graph """
    edges_to_add = []
    # find the room nodes
    room_node_idx = graph['room_node_idx']
    # extract the edges only for the rooms
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in room_node_idx and edge[0] != 0
    ]
    # Get a list (sorted by length) of disconnected components
    sorted_dc = get_dc_comps(room_node_idx, filtered_edges)
    length_of_dc = len(sorted_dc)
    while length_of_dc > 1:
        comps = sorted_dc[0]
        merged_set = set()
        min_cost = 9999
        min_index = -9999
        for s in sorted_dc[1:]:
            merged_set |= s
        for comp in comps:
            for idx, target in enumerate(merged_set):
                cost = math.dist(graph['nodes'][comp][pos],
                                 graph['nodes'][target][pos])
                if cost < min_cost:
                    min_cost = cost
                    min_index = list(merged_set)[idx]

        edge_to_add = (comp, min_index)
        edges_to_add.append(edge_to_add)
        filtered_edges = filtered_edges + [edge_to_add]
        sorted_dc = get_dc_comps(room_node_idx, filtered_edges)
        length_of_dc = len(sorted_dc)

    return edges_to_add


# def euler_to_quaternion(euler):
#     r = R.from_euler('xyz', euler)
#     return r.as_quat()  # [x, y, z, w]


# def quaternion_to_euler(quaternion):
#     r = R.from_quat(quaternion)
#     return r.as_euler('xyz')  # returns roll, pitch, yaw


def get_graph_dict_from_scengraph(graph):
    return {
        'nodes': graph.nodes,
        'edges': graph.edges,
        'edge_index': np.array(graph.edges).T.astype(int),
        'cnt_node_idx': graph.container_indices,
        'obj_node_idx': graph.object_indices,
        'idx_map': graph.asset_id_to_node_idx_map
        }


def get_path_length(path):
    total_length = 0.0
    poses = path.poses

    for i in range(1, len(poses)):
        p0 = poses[i - 1].pose.position
        p1 = poses[i].pose.position
        total_length += math.dist([p0.x, p0.y], [p1.x, p1.y])

    return total_length


def compute_distances(robot_pose, container_nodes):
    distances = {}
    robot_node = {
        'id': 'initial_robot_pose',
        'pos': robot_pose,
    }

    for node1 in [robot_node] + container_nodes:
        for node2 in [robot_node] + container_nodes:
            if node1['id'] == node2['id']:
                distances[(node1['id'] == node2['id'])] = 0.0
                continue
            print(f"Computing distance between {node1['id']} and {node2['id']}")
            
            path = ros_utils.compute_path(node1['pos'], node2['pos'])
            if path is None:
                print("Error computing path!")
            distance = get_path_length(path)
            print(f'Distance={distance}')
            distances[(node1['id'], node2['id'])] = distance
    print(distances)
    return distances






