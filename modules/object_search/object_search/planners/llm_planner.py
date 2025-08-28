from object_search.planners import LearnedPlanner
from object_search.learning.models.llm import GPT, Gemini
from object_search.core import Subgoal
from object_search.learning import utils
from lsp.core import get_frontier_distances, get_robot_distances


class LSPLLMGPTPlanner(LearnedPlanner):
    def __init__(self, target_obj_info, args, destination=None, verbose=True,
                 prompt_template_id=0, fake_llm_response_text=None, use_prompt_caching=True):
        prompt_cache_dir = '/data/.cache/prompt_cache/lsp_llm' if use_prompt_caching else None
        subgoal_property_net = GPT.get_net_eval_fn(prompt_template_id,
                                                   fake_llm_response_text,
                                                   prompt_cache_dir)
        preprocess_input_fn = utils.prepare_lspllm_input
        super(LSPLLMGPTPlanner, self).__init__(target_obj_info,
                                               args,
                                               subgoal_property_net,
                                               preprocess_input_fn,
                                               destination,
                                               verbose)


class LSPLLMGeminiPlanner(LearnedPlanner):
    def __init__(self, target_obj_info, args, destination=None, verbose=True,
                 prompt_template_id=0, fake_llm_response_text=None, use_prompt_caching=True):
        prompt_cache_dir = '/data/.cache/prompt_cache/lsp_llm' if use_prompt_caching else None
        subgoal_property_net = Gemini.get_net_eval_fn(prompt_template_id,
                                                      fake_llm_response_text,
                                                      prompt_cache_dir)
        preprocess_input_fn = utils.prepare_lspllm_input
        super(LSPLLMGeminiPlanner, self).__init__(target_obj_info,
                                                  args,
                                                  subgoal_property_net,
                                                  preprocess_input_fn,
                                                  destination,
                                                  verbose)


class FullLLMPlanner(LearnedPlanner):
    def __init__(self, target_obj_info, args, subgoal_property_net,
                 destination=None, verbose=True):
        super(FullLLMPlanner, self).__init__(target_obj_info,
                                             args,
                                             subgoal_property_net,
                                             self._preprocess_input_fn,
                                             destination,
                                             verbose)
        self.room_distances = None
        self.robot_distances = None

    def _update_subgoal_properties(self):
        # Repurpose this method to compute distances
        # Room distances are computed once since they don't change
        if self.room_distances is None:
            rooms = [Subgoal(idx, self.graph.get_node_position_by_idx(idx)) for idx in self.graph.room_indices]
            room_distances = get_frontier_distances(self.grid, rooms)
            # Convert distances to meters for better interpretability by LLMs
            self.room_distances = {frozenset([r1.id, r2.id]): float(d) * self.args.resolution
                                   for (r1, r2), d in room_distances.items()}

        # Robot distances are computed at every update
        containers = [Subgoal(idx, self.graph.get_node_position_by_idx(idx)) for idx in self.graph.container_indices]
        robot_distances = get_robot_distances(self.grid, self.robot_pose, containers)
        self.robot_distances = {c.id: float(d) * self.args.resolution
                                for c, d in robot_distances.items()}

    def compute_selected_subgoal(self):
        datum = self.preprocess_input_fn(self.graph, self.subgoals, self.target_obj_info)
        chosen_subgoal = self.subgoal_property_net(datum)
        return chosen_subgoal

    def _preprocess_input_fn(self, graph, subgoals, target_obj_info):
        datum = {
            'graph': graph,
            'target_obj_info': target_obj_info,
            'subgoals': subgoals,
            'room_distances': self.room_distances,
            'robot_distances': self.robot_distances,
        }
        return datum


class FullLLMGPTPlanner(FullLLMPlanner):
    def __init__(self, target_obj_info, args, destination=None, verbose=True,
                 prompt_template_id=0, use_prompt_caching=True):
        prompt_cache_dir = '/data/.cache/prompt_cache/full_llm' if use_prompt_caching else None
        subgoal_property_net = GPT.get_search_action_fn(prompt_template_id,
                                                        prompt_cache_dir)
        super(FullLLMGPTPlanner, self).__init__(target_obj_info,
                                                args,
                                                subgoal_property_net,
                                                destination,
                                                verbose)


class FullLLMGeminiPlanner(FullLLMPlanner):
    def __init__(self, target_obj_info, args, destination=None, verbose=True,
                 prompt_template_id=0, use_prompt_caching=True):
        prompt_cache_dir = '/data/.cache/prompt_cache/full_llm' if use_prompt_caching else None
        subgoal_property_net = Gemini.get_search_action_fn(prompt_template_id,
                                                           prompt_cache_dir)
        super(FullLLMGeminiPlanner, self).__init__(target_obj_info,
                                                   args,
                                                   subgoal_property_net,
                                                   destination,
                                                   verbose)
