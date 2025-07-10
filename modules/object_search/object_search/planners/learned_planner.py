# import torch
from .planner import Planner
from object_search.learning import utils
from object_search.learning.models.fcnn import FCNN
from object_search import core
from .planner import Planner

class LearnedPlanner(Planner):
    '''This planner optimistically explores the nearest container to search the target object.'''
    def __init__(self, target_obj_info, args, destination=None, verbose=True):
        super(LearnedPlanner, self).__init__(target_obj_info, args, verbose)
        self.destination = destination

    def compute_selected_subgoal(self):
        thing = self.subgoals[0]
        self.subgoals = self.subgoals[1:]
        return thing