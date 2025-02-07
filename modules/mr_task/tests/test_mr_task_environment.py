import copy
import mr_task
from mr_task.utils import get_inter_distances_nodes
import pouct_planner


def test_mr_task_two_nodes():
    kitchen_node = mr_task.core.Node(name='kitchen', location=(181, 166), is_subgoal=True)
    desk_node = mr_task.core.Node(name='desk', location=(156, 0), is_subgoal=True)
    unk_nodes = [kitchen_node, desk_node]
    robot_nodes = [mr_task.core.RobotNode(mr_task.core.Node(location=(0, 0))) for _ in range(2)]
    node_prop_dict = {
        (kitchen_node, 'Knife'): [1.0, 0, 0],
        (kitchen_node, 'Notebook'): [0.0, 0, 0],
        (desk_node, 'Knife'): [0.0, 0, 0],
        (desk_node, 'Notebook'): [1.0, 0, 0]
    }
    specification = "F Knife & F Notebook"
    distances = get_inter_distances_nodes(unk_nodes, robot_nodes)
    dfa_planner = mr_task.DFAManager(specification)

    mrstate = mr_task.core.MRState(robots=robot_nodes,
                                    planner=copy.copy(dfa_planner),
                                    distances=distances,
                                    subgoal_prop_dict=node_prop_dict,
                                    known_space_nodes=[],
                                    unknown_space_nodes=unk_nodes)

    action, cost, [ordering, costs] = pouct_planner.core.po_mcts(mrstate, n_iterations=50000, C=100)


def test_mr_task_four_nodes():
    shelf_node = mr_task.core.Node(name='shelf', location=(196, 458), props=())
    storage_node = mr_task.core.Node(name='storage', location=(290, 238), props=())
    kitchen_node = mr_task.core.Node(name='kitchen', location=(464, 458), props=('Knife', ))
    desk_node = mr_task.core.Node(name='desk', location=(152, 442), props=('Notebook',))

    ks_nodes = [shelf_node, storage_node, kitchen_node, desk_node]
    robot_nodes = [mr_task.core.RobotNode(mr_task.core.Node(location=(0, 0))) for _ in range(2)]
    distances = get_inter_distances_nodes(ks_nodes, robot_nodes)

    specification = "F Knife & F Notebook"
    dfa_planner = mr_task.DFAManager(specification)
    mrstate = mr_task.core.MRState(robots=robot_nodes,
                                    planner=copy.copy(dfa_planner),
                                    distances=distances,
                                    subgoal_prop_dict={},
                                    known_space_nodes=ks_nodes,
                                    unknown_space_nodes=[])

    # action_desk = mr_task.core.Action(target_node=desk_node)
    # action_shelf = mr_task.core.Action(target_node=shelf_node)
    # action_storage = mr_task.core.Action(target_node=storage_node)
    # action_kitchen = mr_task.core.Action(target_node=kitchen_node)

    def rollout_fn(mrstate):
        return 0
    action, cost, [ordering, costs] = pouct_planner.core.po_mcts(mrstate,
                                                                 n_iterations=50000,
                                                                 C=100,
                                                                 rollout_fn=rollout_fn)
    print([(action.target_node.name, action.props) for action in ordering])
    print(costs)
