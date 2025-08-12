import numpy as np

import procthor
from taskplan import real_world_utils


class DeliveryEnvironment:
    def __init__(self, yaml_file='/resources/worlds/delivery.yaml'):
        self.scenegraph = real_world_utils.utils.get_scene_graph_from_yaml(yaml_file)
        self.graph = real_world_utils.utils.get_graph_dict_from_scengraph(self.scenegraph)
        self.occupancy_grid = self.get_occupancy_grid()
        self.rooms = self.get_rooms()
        self.containers = self.get_containers()
        self.known_cost = self.get_known_cost()
        self.plot_offset = [0, 0]
        self.plot_extent = [0, self.occupancy_grid.shape[0],
                            0, self.occupancy_grid.shape[1]]

    def get_robot_pose(self):
        return (50, 150)  # Decide for left and right

    def get_top_down_frame(self):
        return self.occupancy_grid

    def get_graph(self, include_node_embeddings=True):
        graph = self.graph
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
                procthor.utils.get_sentence_embedding(graph['nodes'][node_key]['name']),
                graph['nodes'][node_key]['type']
            ))
            assert count == node_key
            graph_nodes.append(node_feature)
            node_color_list.append(procthor.utils.get_object_color_from_type(
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

        graph['graph_image'] = procthor.utils.get_graph_image(
            graph['edge_index'],
            node_names, node_color_list
        )
        self.graph = graph
        return graph
    
    def get_rooms(self):
        rooms = [self.scenegraph.nodes[idx] for idx in self.scenegraph.room_indices]
        return rooms

    def get_containers(self):
        containers = [self.scenegraph.nodes[idx] for idx in self.scenegraph.container_indices]
        return containers

    def get_occupancy_grid(self):
        return real_world_utils.ros_utils.get_occupancy_grid()
    
    def get_known_cost(self):
        robot_pose = real_world_utils.ros_utils.get_robot_pose()
        container_nodes = [self.scenegraph.nodes[idx] for idx in self.scenegraph.container_indices]
        return real_world_utils.utils.compute_distances(robot_pose, container_nodes)

