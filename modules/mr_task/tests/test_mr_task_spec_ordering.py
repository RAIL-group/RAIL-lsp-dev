import pytest
import pouct_planner
from mr_task import DFAManager
from mr_task.core import (MRState, Node, Action, advance_mrstate, RobotNode,
                          EventOutcome, History, get_state_from_history)


def test_mr_task_ordered_specification_known():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    ks1_near_foo = Node(props=('foo',), name='s1')
    ks2_bar = Node(props=('bar',), name='s2')
    ks3_far_foo = Node(props=('foo',), name='s3')

    distances = {
        (robot1.start, ks1_near_foo): 5,
        (robot2.start, ks2_bar): 10,
        (ks1_near_foo, ks3_far_foo): 30,
        (ks2_bar, ks3_far_foo): 20,
    }
    action_ks1 = Action(ks1_near_foo)
    action_ks2 = Action(ks2_bar)

    robot1.retarget(action_ks1, distances)
    robot2.retarget(action_ks2, distances)

    specification = "(!foo U bar) & (F foo)"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict={},
                      known_space_nodes=[ks1_near_foo, ks2_bar, ks3_far_foo],
                      unknown_space_nodes=[])

    child_mrstates = advance_mrstate(mrstate)
    assert len(child_mrstates) == 1
    assert child_mrstates[list(child_mrstates.keys())[0]][0] == 1.0
    assert child_mrstates[list(child_mrstates.keys())[0]][1] == 10
    assert list(child_mrstates.keys())[0].planner.has_reached_accepting_state()


def test_mr_task_ordered_specification_unknown_states():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    subgoal_node1 = Node(is_subgoal=True, name='s1')
    subgoal_node2 = Node(is_subgoal=True, name='s2')
    known_space_node_far = Node(props=('foo',), name='s3')

    subgoal_prop_dict = {
        (subgoal_node1, 'foo'): [1.0, 0, 0],
        (subgoal_node1, 'bar'): [0.0, 0, 0],
        (subgoal_node2, 'foo'): [0.0, 0, 0],
        (subgoal_node2, 'bar'): [1.0, 0, 0],
    }

    distances = {
        (robot1.start, subgoal_node1): 5,
        (robot2.start, subgoal_node2): 10,
        (subgoal_node1, known_space_node_far): 30,
        (subgoal_node2, known_space_node_far): 20
    }

    action_unknownS1foo = Action(subgoal_node1, ('foo',), subgoal_prop_dict)
    action_unknownS2bar = Action(subgoal_node2, ('bar',), subgoal_prop_dict)

    robot1.retarget(action_unknownS1foo, distances)
    robot2.retarget(action_unknownS2bar, distances)

    specification = "(!foo U bar) & (F foo)"
    planner = DFAManager(specification)

    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      known_space_nodes=[known_space_node_far],
                      unknown_space_nodes=[subgoal_node1, subgoal_node2])

    child_mrstates = advance_mrstate(mrstate)

    assert len(child_mrstates) == 3
    # foo not found; robot 1 needs to get re-assigned
    not_foo_history = History()
    not_foo_history.add_event(action_unknownS1foo, EventOutcome.FAILURE)
    not_foo_state = get_state_from_history(child_mrstates, not_foo_history)
    assert not not_foo_state.planner.has_reached_accepting_state()
    assert child_mrstates[not_foo_state][0] == 0.0
    assert child_mrstates[not_foo_state][1] == 5

    # foo found, bar found
    history_foo_bar = History()
    history_foo_bar.add_event(action_unknownS1foo, EventOutcome.SUCCESS)
    history_foo_bar.add_event(action_unknownS2bar, EventOutcome.SUCCESS)
    foo_bar_state = get_state_from_history(child_mrstates, history_foo_bar)
    assert foo_bar_state.planner.has_reached_accepting_state()
    assert child_mrstates[foo_bar_state][0] == 1.0
    assert child_mrstates[foo_bar_state][1] == 10

    # foo found, bar not found
    foo_not_bar_history = History()
    foo_not_bar_history.add_event(action_unknownS1foo, EventOutcome.SUCCESS)
    foo_not_bar_history.add_event(action_unknownS2bar, EventOutcome.FAILURE)
    foo_not_bar_state = get_state_from_history(child_mrstates, foo_not_bar_history)
    assert not foo_not_bar_state.planner.has_reached_accepting_state()
    assert child_mrstates[foo_not_bar_state][0] == 0.0
    assert child_mrstates[foo_not_bar_state][1] == 10


def test_mr_task_ordered_specification_known_cost():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    ks1_near_foo = Node(props=('foo',), name='s1')
    ks2_bar = Node(props=('bar',), name='s2')
    ks3_far_foo = Node(props=('foo',), name='s3')

    distances = {
        (robot1.start, ks1_near_foo): 5,
        (robot1.start, ks2_bar): 15,
        (robot1.start, ks3_far_foo): 40,
        (robot2.start, ks1_near_foo): 20,
        (robot2.start, ks2_bar): 10,
        (robot2.start, ks3_far_foo): 30,

        (ks1_near_foo, ks3_far_foo): 30,
        (ks3_far_foo, ks1_near_foo): 30,

        (ks2_bar, ks3_far_foo): 20,
        (ks3_far_foo, ks2_bar): 20,

        (ks1_near_foo, ks2_bar): 15,
        (ks2_bar, ks1_near_foo): 15
    }

    specification = "(!foo U bar) & (F foo)"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict={},
                      known_space_nodes=[ks1_near_foo, ks2_bar, ks3_far_foo],
                      unknown_space_nodes=[])

    def rollout_fn(mrstate):
        return 0

    _, cost, ordering = pouct_planner.core.po_mcts(mrstate, n_iterations=20000, rollout_fn=rollout_fn, C=10.0)
    assert pytest.approx(cost, abs=0.1) == 10
    assert ordering[0][0].target_node == ks1_near_foo
    assert ordering[0][1].target_node == ks2_bar


def test_mr_task_ordered_specification_unknown_cost():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    subgoal_node1 = Node(is_subgoal=True, name='s1')
    subgoal_node2 = Node(is_subgoal=True, name='s2')
    known_space_node_far = Node(props=('foo',), name='s3')

    subgoal_prop_dict = {
        (subgoal_node1, 'foo'): [1.0, 0, 0],
        (subgoal_node1, 'bar'): [0.0, 0, 0],
        (subgoal_node2, 'foo'): [0.0, 0, 0],
        (subgoal_node2, 'bar'): [1.0, 0, 0],
    }

    distances = {
        (robot1.start, subgoal_node1): 5,
        (robot1.start, subgoal_node2): 15,
        (robot1.start, known_space_node_far): 40,
        (robot2.start, subgoal_node1): 20,
        (robot2.start, subgoal_node2): 10,
        (robot2.start, known_space_node_far): 30,
        (subgoal_node1, known_space_node_far): 30,
        (known_space_node_far, subgoal_node1): 30,
        (subgoal_node2, known_space_node_far): 20,
        (known_space_node_far, subgoal_node2): 20,
        (subgoal_node1, subgoal_node2): 15,
        (subgoal_node2, subgoal_node1): 15,
        (subgoal_node1, subgoal_node1): 0,
        (subgoal_node2, subgoal_node2): 0,
    }

    specification = "(!foo U bar) & (F foo)"
    planner = DFAManager(specification)

    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      known_space_nodes=[known_space_node_far],
                      unknown_space_nodes=[subgoal_node1, subgoal_node2])

    def rollout_fn(mrstate):
        return 0
    _, cost, ordering = pouct_planner.core.po_mcts(mrstate, n_iterations=10000, C=1.0, rollout_fn=rollout_fn)
    assert pytest.approx(cost, abs=0.1) == 10
    assert ordering[0][0].target_node == subgoal_node1
    assert ordering[0][1].target_node == subgoal_node2


def test_mr_task_ordered_specification_no_deadlock_happens_unknown():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    subgoal_node1 = Node(is_subgoal=True, name='s1')
    subgoal_node2 = Node(is_subgoal=True, name='s2')
    known_space_node_far = Node(props=('qux',), name='s3')

    subgoal_prop_dict = {
        (subgoal_node1, 'foo'): [1.0, 0, 0],
        (subgoal_node1, 'bar'): [0.0, 0, 0],
        (subgoal_node1, 'qux'): [0.0, 0, 0],
        (subgoal_node2, 'foo'): [0.0, 0, 0],
        (subgoal_node2, 'bar'): [1.0, 0, 0],
        (subgoal_node2, 'qux'): [0.0, 0, 0],
    }

    distances = {
        (robot1.start, subgoal_node1): 5,
        (robot1.start, subgoal_node2): 15,
        (robot1.start, known_space_node_far): 40,
        (robot2.start, subgoal_node1): 20,
        (robot2.start, subgoal_node2): 10,
        (robot2.start, known_space_node_far): 30,
        (subgoal_node1, known_space_node_far): 30,
        (known_space_node_far, subgoal_node1): 30,
        (subgoal_node2, known_space_node_far): 20,
        (known_space_node_far, subgoal_node2): 20,
        (subgoal_node1, subgoal_node2): 15,
        (subgoal_node2, subgoal_node1): 15,
        (subgoal_node1, subgoal_node1): 0,
        (subgoal_node2, subgoal_node2): 0,
    }

    specification = '(!foo U bar) & (!bar U qux) & (F foo)'
    planner = DFAManager(specification)

    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      known_space_nodes=[known_space_node_far],
                      unknown_space_nodes=[subgoal_node1, subgoal_node2,])

    def rollout_fn(mrstate):
        return 0
    _, cost, ordering = pouct_planner.core.po_mcts(mrstate, n_iterations=50000, C=10.0, rollout_fn=rollout_fn)
    assert pytest.approx(cost, abs=0.1) == 45
    assert ordering[0][0].target_node == subgoal_node2
    assert ordering[0][1].target_node == known_space_node_far


def test_mr_task_ordered_specification_no_deadlock_happens_known():
    robot1 = RobotNode(Node())
    robot2 = RobotNode(Node())
    ks_node1 = Node(props=('foo',), name='s1')
    ks_node2 = Node(props=('bar',), name='s2')
    known_space_node_far = Node(props=('qux',), name='s3')

    distances = {
        (robot1.start, ks_node1): 5,
        (robot1.start, ks_node2): 15,
        (robot1.start, known_space_node_far): 40,
        (robot2.start, ks_node1): 20,
        (robot2.start, ks_node2): 10,
        (robot2.start, known_space_node_far): 30,
        (ks_node1, known_space_node_far): 30,
        (known_space_node_far, ks_node1): 30,
        (ks_node2, known_space_node_far): 20,
        (known_space_node_far, ks_node2): 20,
        (ks_node1, ks_node2): 15,
        (ks_node2, ks_node1): 15,
        (ks_node1, ks_node1): 0,
        (ks_node2, ks_node2): 0,
    }

    specification = '(!foo U bar) & (!bar U qux) & (F foo)'
    planner = DFAManager(specification)

    mrstate = MRState(robots=[robot1, robot2],
                      planner=planner,
                      distances=distances,
                      subgoal_prop_dict={},
                      known_space_nodes=[ks_node1, ks_node2, known_space_node_far],
                      unknown_space_nodes=[])


    def rollout_fn(mrstate):
        return 0
    _, cost, ordering = pouct_planner.core.po_mcts(mrstate, n_iterations=50000, C=10.0, rollout_fn=rollout_fn)
    assert pytest.approx(cost, abs=0.1) == 45
    assert ordering[0][0].target_node == ks_node2
    assert ordering[0][1].target_node == known_space_node_far
