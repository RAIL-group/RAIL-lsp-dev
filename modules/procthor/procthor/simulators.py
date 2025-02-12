import copy
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
        # Setup top down camera
        event = self.thor_interface.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        if orthographic:
            pose["orthographicSize"] = 0.5 * max_bound
        else:
            del pose["orthographicSize"]

        # Add the camera to the scene
        event = self.thor_interface.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_image = event.third_party_camera_frames[-1]
        top_down_image = top_down_image[::-1, ...]
        return top_down_image

    def initialize_graph_map_and_subgoals(self):
        random.seed(self.args.current_seed)
        # Select half of the containers as subgoals or at least two
        cnt_count = len(self.known_graph.container_indices)
        lb_sample = min(cnt_count, 2)
        num_of_val_to_choose = max(lb_sample, random.sample(list(range(
            cnt_count // 2, cnt_count)), 1)[0])
        subgoals = random.sample(self.known_graph.container_indices, num_of_val_to_choose)
        for cnt_idx in self.target_obj_info['container_idx']:
            if cnt_idx not in subgoals:
                subgoals.append(cnt_idx)
        subgoals = sorted(subgoals)

        observed_grid = self.known_grid.copy()  # For now, observed grid is the same as known grid
        observed_graph = self.known_graph.get_object_free_graph()

        # Reveal container nodes not chosen as subgoals
        cnt_to_reveal_idx = [xx
                             for xx in self.known_graph.container_indices
                             if xx not in subgoals]
        for node_idx in cnt_to_reveal_idx:
            connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(node_idx, filter_by_type=3)
            for obj_idx in connected_obj_idx:
                o_idx = observed_graph.add_node(self.known_graph.nodes[obj_idx].copy())
                observed_graph.add_edge(node_idx, o_idx)

        return observed_graph, observed_grid, subgoals

    def update_graph_map_and_subgoals(self, observed_graph, observed_grid, subgoals, chosen_subgoal=None):
        if chosen_subgoal is None:
            return observed_graph, observed_grid, subgoals

        subgoals = [s for s in subgoals if s != chosen_subgoal.id]
        graph = observed_graph.copy()

        # Add objects from chosen container to the graph
        connected_obj_idx = self.known_graph.get_adjacent_nodes_idx(chosen_subgoal.id, filter_by_type=3)
        for obj_idx in connected_obj_idx:
            o_idx = graph.add_node(self.known_graph.nodes[obj_idx].copy())
            graph.add_edge(chosen_subgoal.id, o_idx)

        return graph, observed_grid, subgoals
