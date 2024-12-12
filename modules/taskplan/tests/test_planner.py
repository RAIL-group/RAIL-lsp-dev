import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl
import procthor
import taskplan


def get_args():
    args = lambda: None
    args.cache_path = '/data/.cache'
    args.current_seed = 7007
    args.resolution = 0.05
    args.save_dir = '/data/test_logs/'
    args.network_file = '/data/taskplan/logs/dbg/gnn.pt'

    args.cost_type = 'learned'
    args.goal_type = '1object'

    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)

    return args


def test_fast_approximation():
    args = get_args()

    thor_data = procthor.ThorInterface(args=args)

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()

    whole_graph = thor_data.get_graph()
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    learned_data = {
        'partial_map': partial_map,
        'initial_robot_pose': init_robot_pose,
        'learned_net': args.network_file
    }

    t1 = time.time()
    pddl = taskplan.pddl.helper.get_pddl_instance(
        whole_graph=whole_graph,
        map_data=thor_data,
        args=args,
        learned_data=learned_data
    )
    print('PDDL instance generation time:', time.time() - t1)
    raise NotImplementedError
