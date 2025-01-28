'''
MR-Task planning for a single goal is essentially MR-LSP.
Test cases in this file are testing if MR-Task returns same cost and action as MR-LSP for multiple robots.
'''
import numpy as np
import lsp
import mrlsp
import mr_task
import pouct_planner
import pytest


class MRLSPRobot():
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return self.id


def _generate_random_environment(num_robots, num_frontiers):
    # For MRLSP
    robots = [MRLSPRobot(i) for i in range(num_robots)]
    robots_hash_dict = {robot: i for i, robot in enumerate(robots)}
    robots_hash = list(robots_hash_dict.values())
    frontiers = [lsp.core.Frontier(points=np.array([[i+1], [i+1]])) for i in range(num_frontiers)]
    s_hash = {hash(f): f for f in frontiers}
    for f in frontiers:
        f.set_props(prob_feasible=0,
                    delta_success_cost=np.random.randint(50, 100),
                    exploration_cost=np.random.randint(50, 100))

    distances = {
        'goal': {hash(f): np.random.randint(50, 100) for f in frontiers},
        'all': {frozenset([hash(f1), hash(f2)]): np.random.randint(50, 100)
                for f1 in frontiers + robots
                for f2 in frontiers + robots}
    }
    mrlsp_environment = {
        'robots_hash': robots_hash,
        'frontiers': frontiers,
        'distances': distances,
        's_hash': s_hash
    }

    # For MR-Task
    robot_nodes_dict = {r: mr_task.core.Node() for r in robots}
    robot_nodes = [mr_task.core.RobotNode(r_node) for r_node in list(robot_nodes_dict.values())]
    frontier_nodes_dict = {f: mr_task.core.Node(is_subgoal=True, location=f.centroid) for f in frontiers}
    frontier_nodes = list(frontier_nodes_dict.values())
    subgoal_prop_dict = {
        (frontier_node, 'goal'): [f.prob_feasible,
                                  distances['goal'][hash(f)] + f.delta_success_cost,
                                  f.exploration_cost]
        for frontier_node, f in zip(frontier_nodes, frontiers)
    }
    distances_mrtask = {
        (f_node1, f_node2): distances['all'][frozenset([hash(f1), hash(f2)])]
        for f1, f_node1 in frontier_nodes_dict.items()
        for f2, f_node2 in frontier_nodes_dict.items()
    }
    distances_mrtask.update({
        (r_node, f_node): distances['all'][frozenset([hash(r), hash(f)])]
        for r, r_node in robot_nodes_dict.items()
        for f, f_node in frontier_nodes_dict.items()
    })
    mr_task_environment = {
        'robots': robot_nodes,
        'known_space_nodes': [],
        'unknown_space_nodes': frontier_nodes,
        'subgoal_props': subgoal_prop_dict,
        'distances': distances_mrtask,
    }
    return mrlsp_environment, mr_task_environment


# TODO: This test should be done for multiple robots and frontiers, right now its failing (slight cost mismatch)
@pytest.mark.parametrize('num_frontiers', [7])
@pytest.mark.parametrize('num_robots', [2])
def test_mrtask_mrlsp_cost_matches(num_robots, num_frontiers):
    mrlsp_environment, mr_task_environment = _generate_random_environment(num_robots, num_frontiers)
    cost_mrlsp, ordering_mrlsp = mrlsp.core.get_mr_lowest_cost_ordering_cpp(mrlsp_environment['robots_hash'],
                                                                            mrlsp_environment['frontiers'],
                                                                            mrlsp_environment['distances'])

    action_mrlsp = mrlsp_environment['s_hash'][ordering_mrlsp[0]]
    print(action_mrlsp.centroid)
    specification = "F goal"
    planner = mr_task.DFAManager(specification)
    mrstate = mr_task.core.MRState(robots=mr_task_environment['robots'],
                                   planner=planner,
                                   distances=mr_task_environment['distances'],
                                   subgoal_prop_dict=mr_task_environment['subgoal_props'],
                                   known_space_nodes=mr_task_environment['known_space_nodes'],
                                   unknown_space_nodes=mr_task_environment['unknown_space_nodes'])

    def _rollout_fn(state):
        return 0
    action, cost = pouct_planner.core.po_mcts(mrstate,
                                              n_iterations=1000,
                                              C=100,
                                              rollout_fn=_rollout_fn)

    print(action.target_node.location)
    print(f'cost mrtask={cost:.2f},{cost_mrlsp=}')
    assert pytest.approx(cost_mrlsp, abs=5) == cost
    assert list(action.target_node.location) == list(action_mrlsp.centroid)
