import numpy as np

import procthor
from taskplan import real_world_utils

OBJECTS = {
            'fridge': ['waterbottle'],
            'desk': ['cellphone']
        }

class DeliveryEnvironment:
    def __init__(self, yaml_file='/resources/worlds/delivery.yaml'):
        self.scenegraph = real_world_utils.utils.get_scene_graph_from_yaml(yaml_file, OBJECTS)
        # self.add_objects()
        self.graph = real_world_utils.utils.get_graph_dict_from_scengraph(self.scenegraph)
        self.occupancy_grid = self.get_occupancy_grid()
        self.robot_pose = real_world_utils.ros_utils.get_robot_pose()
        self.rooms = self.get_rooms()
        self.containers = self.get_containers()
        self.known_cost = self.get_known_cost()
        self.known_cost_coords = self.get_known_cost_coords()
        self.plot_offset = [0, 0]
        self.plot_extent = [0, self.occupancy_grid.shape[0],
                            0, self.occupancy_grid.shape[1]]
        
    # def add_objects(self):
    #     objects = {
    #         'fridge': ['watebottle'],
    #         'desk': ['cellphone']
    #     }
    #     for i, node in self.scenegraph.nodes.items():
    #         if node['name'] in objects:
    #             node['children'] = objects[node['name']]

    def get_robot_pose(self):
        # return (50, 150)  # Decide for left and right
        return self.robot_pose

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
        # print(f"{graph['edges']=}")
        # print(f"{graph['edge_index']=}")
        # print(f"{graph['graph_edge_index']=}")
        # exit()
        graph['obj_node_idx'] = self.scenegraph.object_indices

        # graph['graph_image'] = procthor.utils.get_graph_image(
        #     graph['edge_index'],
        #     node_names, node_color_list
        # )
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
        robot_pose = self.robot_pose
        container_nodes = [self.scenegraph.nodes[idx] for idx in self.scenegraph.container_indices]
        known_cost = real_world_utils.utils.compute_distances(robot_pose, container_nodes)
        known_cost_s = {}
        for (s , d), v in known_cost.items():
            if s not in known_cost_s:
                known_cost_s[s] = {}
            # known_cost_s = {}
            known_cost_s[s][d] = v

        # print(known_cost_s)
        # exit()
        return known_cost_s
    
    def get_known_cost_coords(self):
        nodes = [self.scenegraph.nodes[i] for i in self.scenegraph.container_indices + self.scenegraph.room_indices]
        cost_dict = real_world_utils.utils.compute_distances(self.robot_pose, nodes)
        cost_coords = {}
        for (id1, id2), v in cost_dict.items():
            if id1 == 'initial_robot_pose':
                coord1 = self.robot_pose
            else:
                idx1 = self.scenegraph.get_node_indices_by_id(id1)[0]
                coord1 = self.scenegraph.nodes[idx1]['position']
            if id2 == 'initial_robot_pose':
                coord2 = self.robot_pose
            else:
                idx2 = self.scenegraph.get_node_indices_by_id(id2)[0]
                coord2 = self.scenegraph.nodes[idx2]['position']
            cost_coords[(coord1, coord2)] = v
        return cost_coords
    
    # def get_distances_lsp(self):
    #     robot_distances = {}
    #     for k, v in self.known_cost['initial_robot_pose'].items():
    #         robot_distances[k] = v
    #     subgoal_distances = {}
    #     container_idx = self.scenegraph.container_indices[0]
    #     container_id = self.scenegraph.nodes[container_idx]['id']
    #     for k, v in self.known_cost[container_id].items():
    #         subgoal_distances[frozenset([container_id, k])] = v
        
    #     goal_distances_all = {}


