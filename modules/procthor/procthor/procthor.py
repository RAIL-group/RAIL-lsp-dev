import json
import copy
import numpy as np
import random
from shapely import geometry
from ai2thor.controller import Controller
from . import utils

IGNORE_CONTAINERS = [
    'baseballbat', 'basketBall', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'basketball', 'box'
]


class SceneGraph:
    def __init__(self):
        """Initialize an empty scene graph."""
        self.nodes = {}  # id -> node dictionary mapping
        self.edges = []  # list of (src, dst) tuples
        self.asset_id_to_node_idx_map = {}  # asset id -> node index mapping

    def add_node(self, node_dict, node_idx=None):
        """Add a new node to the graph."""
        node_idx = len(self.nodes) if node_idx is None else node_idx
        self.nodes[node_idx] = node_dict
        if 'id' in node_dict:
            self.asset_id_to_node_idx_map[node_dict['id']] = node_idx
        return node_idx

    def add_edge(self, src_idx, dst_idx):
        """Add an edge between two nodes by their indices."""
        if src_idx not in self.nodes or dst_idx not in self.nodes:
            raise ValueError('Invalid node indices')
        self.edges.append((src_idx, dst_idx))

    def delete_node(self, node_idx):
        """Delete a node from the graph."""
        if node_idx not in self.nodes:
            raise ValueError('Invalid node index')
        del self.nodes[node_idx]
        self.edges = [(src, dst) for src, dst in self.edges
                      if src != node_idx and dst != node_idx]

    def delete_edge(self, src_idx, dst_idx):
        """Delete an edge between two nodes by their indices."""
        if (src_idx, dst_idx) not in self.edges:
            raise ValueError('Invalid edge')
        self.edges.remove((src_idx, dst_idx))

    def get_node_indices_by_type(self, type_idx):
        """Get indices of all nodes of a given type."""
        return [idx for idx, node in self.nodes.items()
                if node['type'][type_idx] == 1]

    def get_node_indices_by_id(self, id):
        """Get indices of all nodes of a given id."""
        return [idx for idx, node in self.nodes.items()
                if node['id'] == id]

    def get_node_indices_by_name(self, name):
        """Get indices of all nodes of a given name."""
        return [idx for idx, node in self.nodes.items()
                if node['name'] == name]

    def check_if_node_exists_by_id(self, id):
        """Check if a node with a given id exists."""
        return any(node['id'] == id for node in self.nodes.values())

    def get_object_free_graph(self):
        """Get a copy of the graph with object nodes removed."""
        graph = self.copy()
        obj_idx = graph.object_indices
        for _, v in self.edges:
            if v in obj_idx:
                graph.delete_node(v)
        return graph

    def get_node_name_by_idx(self, node_idx):
        """Get name of a node by its index."""
        return self.nodes[node_idx]['name']

    def get_node_position_by_idx(self, node_idx):
        """Get position of a node by its index."""
        return self.nodes[node_idx]['position']

    def __len__(self):
        return len(self.nodes)

    def copy(self):
        """Create a deep copy of the scene graph."""
        graph_copy = SceneGraph()
        graph_copy.nodes = copy.deepcopy(self.nodes)
        graph_copy.edges = copy.deepcopy(self.edges)
        graph_copy.asset_id_to_node_idx_map = copy.deepcopy(self.asset_id_to_node_idx_map)
        return graph_copy

    @property
    def room_indices(self):
        """Get indices of all room nodes."""
        return self.get_node_indices_by_type(1)

    @property
    def container_indices(self):
        """Get indices of all container nodes."""
        return self.get_node_indices_by_type(2)

    @property
    def object_indices(self):
        """Get indices of all object nodes."""
        return self.get_node_indices_by_type(3)

    def get_adjacent_nodes_idx(self, node_idx, filter_by_type=None):
        """Get indices of all adjacent nodes of a given type."""
        adj_nodes_idx = set()
        for src, dst in self.edges:
            if src == node_idx:
                if filter_by_type is None or self.nodes[dst]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(dst)
            elif dst == node_idx:
                if filter_by_type is None or self.nodes[src]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(src)
        return list(adj_nodes_idx)

    def get_parent_node_idx(self, node_idx):
        """Get the index of the parent node of a given node."""
        node_type = self.nodes[node_idx]['type'].index(1)
        parent_nodes_idx = self.get_adjacent_nodes_idx(node_idx, filter_by_type=node_type - 1)
        if len(parent_nodes_idx) == 0:
            return None
        return parent_nodes_idx[0]  # Assuming only one parent node

    def ensure_connectivity(self, occupancy_grid):
        """Ensure the graph is connected by adding necessary edges."""
        required_edges = utils.get_edges_for_connected_graph(occupancy_grid, {
            'nodes': self.nodes,
            'edge_index': self.edges,
            'cnt_node_idx': self.container_indices,
            'obj_node_idx': self.object_indices,
            'idx_map': self.asset_id_to_node_idx_map

        })
        self.edges.extend(required_edges)


class ThorInterface:
    def __init__(self, args, preprocess=True):
        self.args = args
        self.seed = args.current_seed
        self.grid_resolution = args.resolution

        self.scene = self.load_scene()

        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            # prevent adding objects if a container of that type already exists
            container_types = set()
            for container in self.containers:
                container_types.add(container['id'].split('|')[0].lower())
            for container in self.containers:
                filtered_children = []
                if 'children' in container:
                    for child in container['children']:
                        if child['id'].split('|')[0].lower() in container_types:
                            continue
                        filtered_children.append(child)
                    container['children'] = filtered_children
            # filter containers from IGNORE list
            self.containers = [
                container for container in self.containers
                if container['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
            ]

        self.controller = Controller(scene=self.scene,
                                     gridSize=self.grid_resolution,
                                     width=480, height=480)
        # Note: Load occupancy_grid before scene_graph
        self.occupancy_grid = self.get_occupancy_grid()
        self.scene_graph = self.get_scene_graph()
        self.robot_pose = self.get_robot_pose()
        self.target_obj_info = self.get_target_obj_info(self.scene_graph, distinct=False)
        self.known_cost = self.get_known_costs()

    def gen_map_and_poses(self):
        """Generate a map and initial robot poses."""
        return (self.scene_graph,
                self.occupancy_grid,
                self.robot_pose,
                self.target_obj_info)

    def load_scene(self, path='/resources/procthor-10k'):
        with open(
            f'{path}/data.jsonl',
            "r",
        ) as json_file:
            json_list = list(json_file)
        return json.loads(json_list[self.seed])

    def set_grid_offset(self, min_x, min_y):
        self.grid_offset = np.array([min_x, min_y])

    def scale_to_grid(self, point):
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def get_robot_pose(self):
        position = self.agent['position']
        position = np.array([position['x'], position['z']])
        return self.scale_to_grid(position)

    def get_target_obj_info(self, scene_graph, distinct=False):
        random.seed(self.args.current_seed)
        # Get the index of the target object and its name
        target_obj_idx = random.sample(scene_graph.object_indices, 1)[0]
        target_obj_name = scene_graph.nodes[target_obj_idx]['name']
        target_object_type = scene_graph.nodes[target_obj_idx]['type']

        if distinct:
            target_container_idx = scene_graph.get_adjacent_nodes_idx(target_obj_idx, filter_by_type=2)
            return {
                'name': target_obj_name,
                'idx': [target_obj_idx],
                'type': target_object_type,
                'container_idx': target_container_idx
            }

        # Get the indices of all objects with the same name
        all_target_obj_idx = []
        for idx in scene_graph.nodes:
            if scene_graph.nodes[idx]['name'] == target_obj_name:
                all_target_obj_idx.append(idx)
        # Get the indices of the containers containing the target object
        all_target_container_idx = []
        for u, v in scene_graph.edges:
            if v in all_target_obj_idx:
                all_target_container_idx.append(u)
        target_obj_info = {
            'name': target_obj_name,
            'idx': all_target_obj_idx,
            'type': target_object_type,
            'container_idx': all_target_container_idx
        }
        return target_obj_info

    def get_occupancy_grid(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        RPs = reachable_positions

        xs = [rp["x"] for rp in reachable_positions]
        zs = [rp["z"] for rp in reachable_positions]

        # Calculate the mins and maxs
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.set_grid_offset(x_offset, z_offset)

        # Create list of free points
        points = list(zip(xs, zs))
        grid_to_points_map = {self.scale_to_grid(point): RPs[idx]
                              for idx, point in enumerate(points)}
        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height + 2, width + 2), dtype=int)
        free_positions = grid_to_points_map.keys()
        for pos in free_positions:
            occupancy_grid[pos] = 0

        # store the mapping from grid coordinates to simulator positions
        self.g2p_map = grid_to_points_map

        # set the nearest freespace container positions
        for container in self.containers:
            position = container['position']
            if position is not None:
                # get nearest free space pose
                nearest_fp = utils.get_nearest_free_point(position, points)
                # then scale the free space pose to grid
                scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
                # finally set the scaled grid pose as the container position
                container['position'] = scaled_position  # 2d only
                container['id'] = container['id'].lower()  # 2d only

                # next do the same if there is any children of this container
                if 'children' in container:
                    children = container['children']
                    for child in children:
                        child['position'] = container['position']
                        child['id'] = child['id'].lower()

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            point = {'x': point.x, 'z': point.y}
            nearest_fp = utils.get_nearest_free_point(point, points)
            scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
            room['position'] = scaled_position  # 2d only

        return occupancy_grid

    def get_scene_graph(self):
        """Create a scene graph from scene data."""
        graph = SceneGraph()

        # Add apartment node
        apartment_idx = graph.add_node(
            {
                'id': 'Apartment|0',
                'name': 'apartment',
                'position': (0, 0),
                'type': [1, 0, 0, 0]
            }
        )

        # Add room nodes
        for room in self.rooms:
            room_idx = graph.add_node(
                {
                    'id': room['id'],
                    'name': room['roomType'].lower(),
                    'position': room['position'],
                    'type': [0, 1, 0, 0]
                }
            )
            graph.add_edge(apartment_idx, room_idx)

        # Add edges between connected rooms
        room_indices = graph.get_node_indices_by_type(1)
        for i, src_idx in enumerate(room_indices):
            for dst_idx in room_indices[i + 1:]:
                src_node = graph.nodes[src_idx]
                dst_node = graph.nodes[dst_idx]
                if utils.has_edge(self.scene['doors'], src_node['id'], dst_node['id']):
                    graph.add_edge(src_idx, dst_idx)

        # Add container nodes
        for container in self.containers:
            room_id = utils.get_room_id(container['id'])
            room_node_idx = next(idx for idx, node in graph.nodes.items()
                                 if node['type'][1] == 1 and utils.get_room_id(node['id']) == room_id)

            container_idx = graph.add_node(
                {
                    'id': container['id'],
                    'name': utils.get_generic_name(container['id']),
                    'position': container['position'],
                    'type': [0, 0, 1, 0]
                }
            )
            # graph.asset_id_to_node_idx_map[container['id']] = container_idx
            graph.add_edge(room_node_idx, container_idx)

        # Add object nodes for container contents
        for container in self.containers:
            connected_objects = container.get('children')
            if connected_objects is not None:
                container_idx = graph.asset_id_to_node_idx_map[container['id']]
                for obj in connected_objects:
                    # graph.asset_id_to_node_idx_map[obj['id']] = len(graph)
                    obj_idx = graph.add_node(
                        {
                            'id': obj['id'],
                            'name': utils.get_generic_name(obj['id']),
                            'position': obj['position'],
                            'type': [0, 0, 0, 1]
                        }
                    )
                    graph.add_edge(container_idx, obj_idx)

        # Ensure graph connectivity
        graph.ensure_connectivity(self.occupancy_grid)

        return graph

    def get_known_costs(self):
        known_cost = {'initial_robot_pose': {}}
        init_r = self.get_robot_pose()
        cnt_ids = ['initial_robot_pose'] + [cnt['id'] for cnt in self.containers]
        cnt_positions = [init_r] + [cnt['position'] for cnt in self.containers]

        # get cost from one container to another
        for index1, cnt1_id in enumerate(cnt_ids):
            cnt1_position = cnt_positions[index1]
            known_cost[cnt1_id] = {}
            for index2, cnt2_id in enumerate(cnt_ids):
                if cnt2_id not in known_cost:
                    known_cost[cnt2_id] = {}
                if cnt1_id == cnt2_id:
                    known_cost[cnt1_id][cnt2_id] = 0.0
                    continue
                cnt2_position = cnt_positions[index2]
                cost = utils.get_cost(grid=self.occupancy_grid,
                                      robot_pose=cnt1_position,
                                      end=cnt2_position)
                known_cost[cnt1_id][cnt2_id] = round(cost, 4)
                known_cost[cnt2_id][cnt1_id] = round(cost, 4)

        return known_cost
