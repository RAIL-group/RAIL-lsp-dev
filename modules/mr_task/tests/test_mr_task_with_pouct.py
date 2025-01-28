import pytest
import pouct_planner
from mr_task import DFAManager
from mr_task.core import MRState, Node
from mr_task.core import RobotNode


def test_mrtask_pouct_known_cost_action():
    # Set up the environment
    robot_node = Node()
    robot_known = RobotNode(robot_node)
    known_space_node_near = Node(props=('objA', 'objB',), location=(5, 0))
    known_space_node_far = Node(props=('objA', 'objB',), location=(100, 0))

    distances = {
        (robot_node, known_space_node_near): 5,
        (robot_node, known_space_node_far): 100
    }
    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_known],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict={},
                      known_space_nodes=[known_space_node_near, known_space_node_far],
                      unknown_space_nodes=[])

    best_action, cost = pouct_planner.core.po_mcts(mrstate, n_iterations=10000, C=10.0)
    assert cost == 5
    assert best_action.target_node == known_space_node_near


def test_mrtask_pouct_single_robot_goal_two_subgoals():
    robot_node = Node()
    robot = RobotNode(robot_node)
    subgoal_node1 = Node(is_subgoal=True, location='a')
    subgoal_node2 = Node(is_subgoal=True, location='b')

    subgoal_prop_dict = {
        (subgoal_node1, 'goal'): [0.2, 10, 20],
        (subgoal_node2, 'goal'): [0.9, 30, 50]
    }
    distances = {
        (robot_node, subgoal_node1): 30,
        (robot_node, subgoal_node2): 30,
        (subgoal_node1, subgoal_node2): 20, (subgoal_node2, subgoal_node1): 20
    }
    specification = "F goal"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      known_space_nodes=[],
                      unknown_space_nodes=[subgoal_node1, subgoal_node2])

    obtained_action, obtained_cost = pouct_planner.core.po_mcts(mrstate,
                                                                n_iterations=10000, C=10.0)
    node1, cost1 = subgoal_node1, 30 + 10 + 0.8 * (10 + 20 + 30)
    node2, cost2 = subgoal_node2, 30 + 30 + 0.1 * (30 + 20 + 10)
    actual_node, actual_cost = (node1, cost1) if cost1 < cost2 else (node2, cost2)
    assert obtained_action.target_node == actual_node
    assert pytest.approx(obtained_cost, abs=1.0) == actual_cost
    assert obtained_action.target_node == actual_node


def test_mrtask_pouct_single_goal_two_robots_subgoals():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    subgoal_node1 = Node(is_subgoal=True, location='a')
    subgoal_node2 = Node(is_subgoal=True, location='b')

    subgoal_prop_dict = {
        (subgoal_node1, 'goal'): [0.2, 10, 20],
        (subgoal_node2, 'goal'): [0.9, 30, 50]
    }
    distances = {
        (robot1.start, subgoal_node1): 30,
        (robot1.start, subgoal_node2): 30,
        (robot2.start, subgoal_node1): 30,
        (robot2.start, subgoal_node2): 30,
        (subgoal_node1, subgoal_node2): 20, (subgoal_node2, subgoal_node1): 20,
        (subgoal_node1, subgoal_node1): 0, (subgoal_node2, subgoal_node2): 0
    }
    specification = "F goal"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      known_space_nodes=[],
                      unknown_space_nodes=[subgoal_node1, subgoal_node2])

    obtained_action, obtained_cost = pouct_planner.core.po_mcts(
        mrstate, n_iterations=10000, C=10.0)

    assert pytest.approx(obtained_cost, abs=1.0) == (0.2 * 40 + 0.8 * 60)
