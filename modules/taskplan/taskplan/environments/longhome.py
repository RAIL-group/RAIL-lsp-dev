import numpy as np
from procthor import utils


class LongHome:
    def __init__(self):
        self.occupancy_grid = get_occupancy_grid()
        self.rooms = get_rooms()
        self.containers = get_objects()
        self.known_cost = get_known_cost(self.occupancy_grid, self.containers,
                                         self.get_robot_pose())

        self.plot_offset = [0, 0]
        self.plot_extent = [0, self.occupancy_grid.shape[0], 0, self.occupancy_grid.shape[1]]

    def get_robot_pose(self):
        return (800, 150)

    def get_top_down_frame(self):
        return self.occupancy_grid.T

    def get_graph(self, include_node_embeddings=True):
        # This method creates graph data from custom data
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

        for room in self.rooms:
            nodes[node_count] = {
                'id': room['id'],
                'name': room['roomType'].lower(),
                'pos': room['position'],
                'type': [0, 1, 0, 0]
            }
            edges.append(tuple([0, node_count]))
            node_count += 1

        room_edges = set()
        if len(self.rooms) > 1:
            room_edges.add((1, 2))
            room_edges.add((2, 3))
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


def get_occupancy_grid():
    # Define the size of the grid
    grid_width = 900
    grid_height = 300
    occupied = 1

    # Create a grid initialized with freespace value (0)
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Define the positions and sizes of the objects in the environment
    # Coordinates are (x, y), with origin (0, 0) at the top-left of the grid

    # First the kitchen side of the environment
    # Sink
    set_rectangle(grid, (10, 10), (30, 50), occupied)

    # Countertop
    set_rectangle(grid, (10, 230), (50, 290), occupied)

    # Dining table
    set_rectangle(grid, (160, 250), (190, 290), occupied)

    # Set First wall
    set_rectangle(grid, (200, 0), (210, grid_height//2-20), occupied)
    set_rectangle(grid, (200, grid_height//2+20), (210, grid_height), occupied)

    # Second the bedroom
    # bed
    set_rectangle(grid, (260+300, 250), (320+300, 290), occupied)
    # dresser
    set_rectangle(grid, (370+300, 50), (390+300, 100), occupied)

    # Set Second wall
    set_rectangle(grid, (400+300, 0), (410+300, grid_height//2-20), occupied)
    set_rectangle(grid, (400+300, grid_height//2+20), (410+300, grid_height), occupied)

    # shelf
    set_rectangle(grid, (420+300, 270), (460+300, 290), occupied)
    # table
    set_rectangle(grid, (480+300, 270), (520+300, 290), occupied)
    # table
    set_rectangle(grid, (850, 140), (870, 160), occupied)
    # side table
    set_rectangle(grid, (720, 60), (760, 80), occupied)

    # # Two chairs near the table
    # set_rectangle(grid, (90, 70), (100, 90), occupied)  # Chair 1
    # set_rectangle(grid, (140, 70), (150, 90), occupied)  # Chair 2

    return grid.T


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


def get_rooms():
    kitchen = {
        'id': 'kitchen|0|1',
        'roomType': 'kitchen',
        'position': (100, 150)  # (150, 100)
    }
    bedroom = {
        'id': 'bedroom|0|2',
        'roomType': 'bedroom',
        'position': (300, 150)  # (150, 300)
    }
    livingroom = {
        'id': 'livingroom|0|3',
        'roomType': 'livingroom',
        'position': (500+300, 150)  # (150, 500)
    }
    rooms = [kitchen, bedroom, livingroom]
    return rooms


def get_objects():
    # list the objects bread, toaster, plate, coffeegrinds, coffeemachine, mug, waterbottle
    bread = {'id': 'bread|0|1'}
    toaster = {'id': 'toaster|0|1'}
    plate = {'id': 'plate|0|1'}
    coffeegrinds = {'id': 'coffeegrinds|0|1'}
    coffeemachine = {'id': 'coffeemachine|0|1'}
    mug = {'id': 'mug|0|1'}
    waterbottle = {'id': 'waterbottle|0|1'}

    # list the containers where objects can be found
    sink = {
        'id': 'sink|1|0',
        'position': (31, 51),
        'children': []
    }
    countertop = {
        'id': 'countertop|1|0',
        'position': (51, 260),
        'children': [toaster, bread, plate]
    }
    garbagecan = {
        'id': 'garbagecan|1|0',
        'position': (175, 249),
        'children': []
    }
    dryingrack = {
        'id': 'dryingrack|2|0',
        'position': (380+300, 101)
    }
    bed = {
        'id': 'bed|2|0',
        'position': (290+300, 249)
    }
    diningtable = {
        'id': 'diningtable|3|0',
        'position': (440+300, 269),
        'children': [waterbottle]
    }
    coffeetable = {
        'id': 'coffeetable|3|0',
        'position': (500+300, 269),
        'children': [coffeegrinds, coffeemachine]
    }
    desk = {
        'id': 'desk|3|0',
        'position': (849, 150),
        'children': []
    }
    sidetable = {
        'id': 'sidetable|3|0',
        'position': (740, 81),
        'children': [mug]
    }
    objects = [sink, countertop, garbagecan, dryingrack, bed,
               diningtable, coffeetable, desk, sidetable]
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
