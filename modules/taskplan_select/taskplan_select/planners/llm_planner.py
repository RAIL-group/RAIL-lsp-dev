from taskplan.planners.planner import LearnedPlanner
from taskplan_select.learning.models.llm import GPT4, Gemini
from taskplan.core import Subgoal, get_subgoal_distances, get_robot_distances


class LSPLLMPlanner(LearnedPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True, destination=None):
        super(LearnedPlanner, self).__init__(
            target_obj_info, args, device, verbose)
        self.destination = destination
        self.subgoal_property_net = None

    def _update_subgoal_properties(self):
        datum = {
            'graph': self.graph,
            'target_obj_info': self.target_obj_info,
        }
        for subgoal in self.subgoals:
            datum['subgoal'] = subgoal
            prob_feasible = self.subgoal_property_net(datum)
            subgoal.set_props(prob_feasible=prob_feasible)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:6.4f}|'
                    f'at= {self.graph.get_node_name_by_idx(subgoal.id)}'
                )


class LSPLLMGPT4Planner(LSPLLMPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None, prompt_template_id=0, fake_llm_response_text=None, use_prompt_caching=True):
        super(LSPLLMGPT4Planner, self).__init__(target_obj_info, args, device, verbose, destination)
        self.subgoal_property_net = GPT4.get_net_eval_fn(prompt_template_id,
                                                         fake_llm_response_text,
                                                         use_prompt_caching)


class LSPLLMGeminiPlanner(LSPLLMPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None, prompt_template_id=0, fake_llm_response_text=None, use_prompt_caching=True):
        super(LSPLLMGeminiPlanner, self).__init__(target_obj_info, args, device, verbose, destination)
        self.subgoal_property_net = Gemini.get_net_eval_fn(prompt_template_id,
                                                           fake_llm_response_text,
                                                           use_prompt_caching)


class FullLLMPlanner(LearnedPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None):
        super(LearnedPlanner, self).__init__(
            target_obj_info, args, device, verbose)
        self.destination = destination
        self.subgoal_property_net = None
        self.room_distances = None
        self.robot_distances = None

    def _update_subgoal_properties(self):
        # Repurpose this method to compute distances
        # Room distances are computed once since they don't change
        if self.room_distances is None:
            rooms = [Subgoal(idx, self.graph.get_node_position_by_idx(idx)) for idx in self.graph.room_indices]
            room_distances = get_subgoal_distances(self.grid, rooms)
            # Convert distances to meters for better interpretability by LLMs
            self.room_distances = {frozenset([r1.id, r2.id]): float(d) * self.args.resolution
                                   for (r1, r2), d in room_distances.items()}

        # Robot distances are computed at every update
        containers = [Subgoal(idx, self.graph.get_node_position_by_idx(idx)) for idx in self.graph.container_indices]
        robot_distances = get_robot_distances(self.grid, self.robot_pose, containers)
        self.robot_distances = {c.id: float(d) * self.args.resolution
                                for c, d in robot_distances.items()}

    def compute_selected_subgoal(self):
        datum = {
            'graph': self.graph,
            'target_obj_info': self.target_obj_info,
            'subgoals': self.subgoals,
            'room_distances': self.room_distances,
            'robot_distances': self.robot_distances,
        }
        chosen_subgoal = self.subgoal_property_net(datum)
        return chosen_subgoal


class FullLLMGPT4Planner(FullLLMPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None, prompt_template_id=0, use_prompt_caching=True):
        super(FullLLMGPT4Planner, self).__init__(target_obj_info, args, device, verbose, destination)
        self.subgoal_property_net = GPT4.get_search_action_fn(prompt_template_id,
                                                              use_prompt_caching)


class FullLLMGeminiPlanner(FullLLMPlanner):
    def __init__(self, target_obj_info, args, device=None, verbose=True,
                 destination=None, prompt_template_id=0, use_prompt_caching=True):
        super(FullLLMGeminiPlanner, self).__init__(target_obj_info, args, device, verbose, destination)
        self.subgoal_property_net = Gemini.get_search_action_fn(prompt_template_id,
                                                                use_prompt_caching)
