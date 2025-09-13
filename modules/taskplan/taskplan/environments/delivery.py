import numpy as np
import json

from procthor import utils
from procthor.scenegraph import SceneGraph


class DeliveryEnvironment:
    def __init__(self):
        self.occupancy_grid = get_occupancy_grid()
        self.rooms = get_rooms()
        self.containers = get_objects()
        # self.known_cost = get_known_cost(self.occupancy_grid, self.containers,
        #                                  self.get_robot_pose())
        self.known_cost, self.container_distances, self.find_cost = get_real_cost()
        self.plot_offset = [0, 0]
        self.plot_extent = [0, self.occupancy_grid.shape[0],
                            0, self.occupancy_grid.shape[1]]

        self.scenegraph = SceneGraph()
        graph = self.get_graph(include_node_embeddings=False)
        self.scenegraph.nodes = graph['nodes']
        self.scenegraph.edges = graph['edge_index']
        self.scenegraph.asset_id_to_node_idx_map = graph['idx_map']
        for key, value in self.scenegraph.asset_id_to_node_idx_map.items():
            print(self.scenegraph.nodes[value])

        self.known_cost_coords = {}
        for (key1, key2), d in self.container_distances.items():
            if key1 == 'initial_robot_pose':
                c1_position = self.get_robot_pose()
            else:
                c1_position = self.scenegraph.nodes[self.scenegraph.asset_id_to_node_idx_map[key1]]['position']
            if key2 == 'initial_robot_pose':
                c2_position = self.get_robot_pose()
            else:
                c2_position = self.scenegraph.nodes[self.scenegraph.asset_id_to_node_idx_map[key2]]['position']
            self.known_cost_coords[(c1_position, c2_position)] = d
            self.known_cost_coords[(c2_position, c1_position)] = d

    def get_robot_pose(self):
        return (430, 150)  # Decide for left and right
        # return (31, 190)  # for the real environment, start in front of fridge
        # return (-4.485677650429344, 1.132334096599578)

    def get_top_down_frame(self):
        return self.occupancy_grid

    def get_graph(self, include_node_embeddings=True):
        ''' This method creates graph data from custom data'''

        # Create dummy apartment node
        node_count = 0
        nodes = {}
        assetId_idx_map = {}
        edges = []
        nodes[node_count] = {
            'id': 'Apartment|0',
            'name': 'apartment',
            'pos': (0, 0),
            'position': (0, 0),
            'type': [1, 0, 0, 0]
        }
        node_count += 1

        # Iterate over rooms but skip position coordinate scaling since not
        # required in distance calculations
        for room in self.rooms:
            nodes[node_count] = {
                'id': room['id'],
                'name': room['roomType'].lower(),
                'pos': room['position'],
                'position': room['position'],
                'type': [0, 1, 0, 0]
            }
            assetId_idx_map[room['id']] = node_count
            edges.append(tuple([0, node_count]))
            node_count += 1

        # manually add an edge between two rooms if the second one exists
        room_edges = set()
        if len(self.rooms) > 1:
            room_edges.add((1, 2))
            # room_edges.add((2, 3))
        edges.extend(room_edges)

        node_keys = list(nodes.keys())
        node_ids = [utils.get_room_id(nodes[key]['id']) for key in node_keys]
        cnt_node_idx = []

        for container in self.containers:
            assetId = container['id']
            room_id = utils.get_room_id(assetId)
            assetId_idx_map[assetId] = node_count
            name = utils.get_generic_name(assetId)
            nodes[node_count] = {
                'id': assetId,
                'name': name,
                'pos': container['position'],
                'position': container['position'],
                'type': [0, 0, 1, 0]
            }
            edges.append(tuple([room_id, node_count]))
            cnt_node_idx.append(node_count)
            node_count += 1

        node_keys = list(nodes.keys())
        node_ids = [nodes[key]['id'] for key in node_keys]
        obj_node_idx = []

        for container in self.containers:
            connected_objects = container.get('children')
            if connected_objects is not None:
                src = node_ids.index(container['id'])
                for object in connected_objects:
                    assetId = object['id']
                    assetId_idx_map[assetId] = node_count
                    name = utils.get_generic_name(assetId)
                    nodes[node_count] = {
                        'id': assetId,
                        'name': name,
                        'pos': container['position'],
                        'position': container['position'],
                        'type': [0, 0, 0, 1]
                    }
                    edges.append(tuple([src, node_count]))
                    obj_node_idx.append(node_count)
                    node_count += 1

        graph = {
            'nodes': nodes,  # dictionary {id, name, pos, type}
            'edge_index': edges,  # pairwise edge list
            'cnt_node_idx': cnt_node_idx,  # indices of contianers
            'obj_node_idx': obj_node_idx,  # indices of objects
            'idx_map': assetId_idx_map  # mapping from assedId to graph index position
        }

        if not include_node_embeddings:
            return graph

        # perform some more formatting for the graph, then return
        node_coords = {}
        node_names = {}
        graph_nodes = []
        node_color_list = []

        for count, node_key in enumerate(graph['nodes']):
            node_coords[node_key] = graph['nodes'][node_key]['pos']
            node_names[node_key] = graph['nodes'][node_key]['name']
            node_feature = np.concatenate((
                utils.get_sentence_embedding(graph['nodes'][node_key]['name']),
                graph['nodes'][node_key]['type']
            ))
            assert count == node_key
            graph_nodes.append(node_feature)
            node_color_list.append(utils.get_object_color_from_type(
                graph['nodes'][node_key]['type']))

        graph['node_coords'] = node_coords
        graph['node_names'] = node_names
        graph['graph_nodes'] = graph_nodes  # node features
        src = []
        dst = []
        for edge in graph['edge_index']:
            src.append(edge[0])
            dst.append(edge[1])
        graph['graph_edge_index'] = [src, dst]

        graph['graph_image'] = utils.get_graph_image(
            graph['edge_index'],
            node_names, node_color_list
        )

        return graph


# Helper function to set rectangular areas as occupied
def set_rectangle(grid, top_left, bottom_right, value):
    """
    Set a rectangular area in the grid as occupied.

    :param grid: The 2D numpy array representing the grid.
    :param top_left: (x, y) coordinate of the top-left corner of the rectangle.
    :param bottom_right: (x, y) coordinate of the bottom-right corner of the rectangle.
    :param value: Value to set in the rectangle.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    grid[y1:y2, x1:x2] = value


def get_occupancy_grid():
    # Define the size of the grid
    grid_width = 1100
    grid_height = 300
    occupied = 1

    # Create a grid initialized with freespace value (0)
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Define the positions and sizes of the objects in the environment
    # Coordinates are (x, y), with origin (0, 0) at the top-left of the grid

    # Dining table
    set_rectangle(grid, (10, 10), (30, 50), occupied)

    # Fridge
    set_rectangle(grid, (10, 180), (30, 200), occupied)

    # Countertop
    set_rectangle(grid, (10, 270), (50, 290), occupied)

    # sink
    set_rectangle(grid, (70, 270), (90, 290), occupied)

    # Garbage can
    set_rectangle(grid, (360, 10), (380, 20), occupied)

    # wall1
    set_rectangle(grid, (400, 0), (410, 110), occupied)
    set_rectangle(grid, (400, 190), (410, 300), occupied)

    # wall2
    set_rectangle(grid, (800, 0), (810, 110), occupied)
    set_rectangle(grid, (800, 190), (810, 300), occupied)

    # bed
    set_rectangle(grid, (820, 10), (840, 50), occupied)

    # tvstand
    set_rectangle(grid, (820, 270), (850, 290), occupied)

    # sofa
    set_rectangle(grid, (990, 70), (1000, 90), occupied)

    # desk
    set_rectangle(grid, (880, 270), (920, 290), occupied)

    return grid.T


def get_rooms():
    kitchen = {
        'id': 'kitchen|1|0',
        'roomType': 'kitchen',
        'position': (200, 150)
    }
    # livingroom = {
    #     'id': 'livingroom|0|2',
    #     'roomType': 'livingroom',
    #     'position': (600, 150)
    # }
    bedroom = {
        'id': 'bedroom|2|0',
        'roomType': 'bedroom',
        'position': (950, 150)
    }
    # rooms = [kitchen, livingroom, bedroom]
    rooms = [kitchen, bedroom]
    return rooms


def get_objects():
    # list the objects remotecontrol and waterbottle
    cellphone = {'id': 'cellphone|2|0'}
    waterbottle = {'id': 'waterbottle|1|0'}
    # newspaper = {'id': 'newspaper|3|0'}
    # apple = {'id': 'apple|1|0'}
    # remote = {'id': 'remotecontrol|2|0'}
    # television = {'id': 'television|2|0'}

    # list the containers that are in kitchen: diningtable, fridge
    # countertop, sink, garbagecan
    diningtable = {
        'id': 'diningtable|1|0',
        'position': (30, 51)
    }
    fridge = {
        'id': 'fridge|1|0',
        'position': (31, 190),
        'children': [waterbottle]
    }
    countertop = {
        'id': 'countertop|1|0',
        'position': (30, 269),
        # 'children': [apple]
    }
    sink = {
        'id': 'sink|1|0',
        'position': (80, 269)
    }
    garbagecan = {
        'id': 'garbagecan|1|0',
        'position': (370, 21)
    }

    # list the containers that are in bedroom:
    # bed, tvstand, sofa, desk
    bed = {
        'id': 'bed|2|0',
        'position': (841, 51)
    }
    tvstand = {
        'id': 'tvstand|2|0',
        'position': (835, 269),
        # 'children': [remote, television]
    }
    sofa = {
        'id': 'couch|2|0',
        'position': (989, 80),
        # 'children': [newspaper]
    }
    desk = {
        'id': 'desk|2|0',
        'position': (879, 280),
        'children': [cellphone]
    }

    objects = [diningtable, fridge, countertop, sink, garbagecan,
               bed, tvstand, sofa, desk]
    return objects


def get_known_cost(occupancy_grid, containers, robot_pose):
    known_cost = {'initial_robot_pose': {}}
    init_r = robot_pose
    cnt_ids = ['initial_robot_pose'] + [cnt['id'] for cnt in containers]
    cnt_positions = [init_r] + [cnt['position'] for cnt in containers]

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
            cost = utils.get_cost(grid=occupancy_grid,
                                  robot_pose=cnt1_position,
                                  end=cnt2_position)
            known_cost[cnt1_id][cnt2_id] = round(cost, 4)
            known_cost[cnt2_id][cnt1_id] = round(cost, 4)

    return known_cost


def get_real_cost(json_file="/data/test_logs/spot_init_fluents_renamed.json"):
    with open(json_file) as f:
        data = json.load(f)

    container_distances = {}
    find_costs = {}

    for key_str, dist in data.items():
        # Convert string representation of tuple back into a real tuple
        key = eval(key_str)  # safe here because keys are tuples
        if key[0] in ("known-cost", "find-cost"):
            if key[0] == "known-cost":
                container_distances[(key[1], key[2])] = dist
            elif key[0] == "find-cost":
                find_costs[(key[1], key[2], key[3])] = dist

    known_cost = {}
    # Ensure symmetry in container distances
    for key in container_distances:
        if key[0] not in known_cost:
            known_cost[key[0]] = {}
            known_cost[key[0]][key[0]] = 0.0
        if key[1] not in known_cost:
            known_cost[key[1]] = {}
            known_cost[key[1]][key[1]] = 0.0
        known_cost[key[0]][key[1]] = container_distances[key]
        known_cost[key[1]][key[0]] = container_distances[key]
    # room_names = ['kitchen|0|1', 'bedroom|0|2']
    # container_distances[(room_names[0], room_names[1])] = 6.5
    # container_distances[(room_names[1], room_names[0])] = 6.5
    # print("Known cost:", known_cost)
    # print("==============")
    return known_cost, container_distances, find_costs
