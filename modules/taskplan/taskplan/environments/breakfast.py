import numpy as np

from procthor import utils


class Breakfast:
    def __init__(self):
        self.occupancy_grid = get_occupancy_grid()
        self.rooms = get_rooms()
        self.containers = get_objects()
        self.known_cost = get_known_cost(self.occupancy_grid, self.containers,
                                         self.get_robot_pose())

    def get_robot_pose(self):
        return (200, 125)

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
                'type': [0, 1, 0, 0]
            }
            edges.append(tuple([0, node_count]))
            node_count += 1

        # manually add an edge between two rooms if the second one exists
        room_edges = set()
        if len(self.rooms) > 1:
            room_edges.add((1, 2))
        # for i in range(1, len(nodes)):
        #     for j in range(i + 1, len(nodes)):
        #         node_1, node_2 = nodes[i], nodes[j]
        #         if utils.has_edge(self.scene['doors'], node_1['id'], node_2['id']):
        edges.extend(room_edges)

        node_keys = list(nodes.keys())
        node_ids = [utils.get_room_id(nodes[key]['id']) for key in node_keys]
        cnt_node_idx = []

        for container in self.containers:
            cnt_id = utils.get_room_id(container['id'])
            src = node_ids.index(cnt_id)
            assetId = container['id']
            assetId_idx_map[assetId] = node_count
            name = utils.get_generic_name(container['id'])
            nodes[node_count] = {
                'id': container['id'],
                'name': name,
                'pos': container['position'],
                'type': [0, 0, 1, 0]
            }
            edges.append(tuple([src, node_count]))
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
                    name = utils.get_generic_name(object['id'])
                    nodes[node_count] = {
                        'id': object['id'],
                        'name': name,
                        'pos': object['position'],
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
    grid_width = 300
    grid_height = 200
    occupied = 1

    # Create a grid initialized with freespace value (0)
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Define the positions and sizes of the objects in the environment
    # Coordinates are (x, y), with origin (0, 0) at the top-left of the grid

    # Dining table
    set_rectangle(grid, (100, 70), (140, 110), occupied)

    # Fridge
    set_rectangle(grid, (10, 150), (30, 190), occupied)

    # Countertop
    set_rectangle(grid, (230, 10), (290, 50), occupied)

    # Shelving unit
    set_rectangle(grid, (10, 10), (30, 50), occupied)

    # Garbage can
    set_rectangle(grid, (270, 180), (290, 200), occupied)

    # Two chairs near the table
    set_rectangle(grid, (90, 70), (100, 90), occupied)  # Chair 1
    set_rectangle(grid, (140, 70), (150, 90), occupied)  # Chair 2

    return grid.T


def get_rooms():
    rooms = []
    room = {
        'id': 'kitchen|0|0',
        'roomType': 'kitchen',
        'position': (150, 100)
    }
    rooms.append(room)
    return rooms


def get_objects():
    # list the objects apple, egg, kettle, plate on the containers
    apple = {
        'id': 'apple|0|0',
        'position': (31, 171)
    }
    egg = {
        'id': 'egg|0|0',
        'position': (31, 171)
    }
    kettle = {
        'id': 'kettle|0|0',
        'position': (260, 51)
    }
    plate = {
        'id': 'plate|0|0',
        'position': (31, 30)
    }

    # list the containers that are: diningtable, fridge, countertop,
    # shelvingunit, garbagecan, chair1, chair2
    diningtable = {
        'id': 'diningtable|0|0',
        'position': (120, 111),
        'children': []
    }
    fridge = {
        'id': 'fridge|0|0',
        'position': (31, 171),
        'children': [apple, egg]
    }
    countertop = {
        'id': 'countertop|0|0',
        'position': (260, 51),
        'children': [kettle]
    }
    shelvingunit = {
        'id': 'shelvingunit|0|0',
        'position': (31, 30),
        'children': [plate]
    }
    garbagecan = {
        'id': 'garbagecan|0|0',
        'position': (280, 178),
        'children': []
    }
    chair1 = {
        'id': 'chair1|0|0',
        'position': (88, 80),
        'children': []
    }
    chair2 = {
        'id': 'chair2|0|0',
        'position': (151, 80),
        'children': []
    }
    objects = [diningtable, fridge, countertop, shelvingunit, garbagecan, chair1, chair2]
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


