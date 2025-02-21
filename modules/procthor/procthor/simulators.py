import random


class SceneGraphSimulator:
    def __init__(self,
                 known_graph,
                 args,
                 target_obj_info,
                 known_grid=None,
                 thor_interface=None,
                 verbose=True):
        self.known_graph = known_graph
        self.args = args
        self.target_obj_info = target_obj_info
        self.known_grid = known_grid
        self.thor_interface = thor_interface
        self.verbose = verbose

    def get_top_down_image(self, orthographic=True):
        if self.thor_interface is None:
            raise ValueError("Thor Interface is not set")

        return self.thor_interface.get_top_down_image(orthographic=orthographic)

    def initialize_graph_and_containers(self):
        random.seed(self.args.current_seed)
        # Select half of the containers as subgoals or at least two
        cnt_count = len(self.known_graph.container_indices)
        lb_sample = min(cnt_count, 2)
        num_of_val_to_choose = max(lb_sample, random.sample(list(range(
            cnt_count // 2, cnt_count)), 1)[0])
        unexplored_containers = random.sample(self.known_graph.container_indices, num_of_val_to_choose)
        for cnt_idx in self.target_obj_info['container_idx']:
            if cnt_idx not in unexplored_containers:
                unexplored_containers.append(cnt_idx)
        unexplored_containers = sorted(unexplored_containers)

        observed_graph = self.known_graph.get_object_free_graph()

        # Reveal container nodes not chosen as subgoals
        cnt_to_reveal_idx = [xx
                             for xx in self.known_graph.container_indices
                             if xx not in unexplored_containers]
        for node_idx in cnt_to_reveal_idx:
            connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(node_idx, filter_by_type=3)
            for obj_idx in connected_obj_idx:
                o_idx = observed_graph.add_node(self.known_graph.nodes[obj_idx].copy())
                observed_graph.add_edge(node_idx, o_idx)

        return observed_graph, unexplored_containers

    def update_graph_and_containers(self, observed_graph, containers, chosen_container_idx=None):
        if chosen_container_idx is None:
            return observed_graph, containers

        unexplored_containers = [s for s in containers if s != chosen_container_idx]
        graph = observed_graph.copy()

        # Add objects from chosen container to the graph
        connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(chosen_container_idx, filter_by_type=3)
        for obj_idx in connected_obj_idx:
            o_idx = graph.add_node(self.known_graph.nodes[obj_idx].copy())
            graph.add_edge(chosen_container_idx, o_idx)

        return graph, unexplored_containers
