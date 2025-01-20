import pytest
from mr_task.core import (Node,
                          Action,
                          History,
                          EventOutcome,
                          get_next_event_and_time)
from mr_task.robot import Robot


def test_mrtask_event_outcome_known():
    # Set up the environment
    robot_node = Node()
    known_space_node = Node(props=('objA', 'objB',))
    action = Action(known_space_node)
    robot = Robot(robot_node)
    distances = {(robot_node, known_space_node): 18}
    robot.retarget(action, distances)

    assert robot.time_remaining >= distances[(robot_node, known_space_node)]
    assert robot.info_time >= distances[(robot_node, known_space_node)]

    # Known space nodes always succeed and with the time it takes to reach it
    history_none = History()
    event_outcome, event_time = get_next_event_and_time(robot, history_none)
    assert event_outcome == EventOutcome.SUCCESS
    assert event_time == distances[(robot_node, known_space_node)]


@pytest.mark.parametrize('target_object', ['objA', 'objB'])
def test_mrtask_event_outcome_subgoal(target_object):
    # Set up the environment
    robot_node = Node()
    robot = Robot(robot_node)
    subgoal_node = Node(is_subgoal=True)
    subgoal_prop_dict = {
        (subgoal_node, 'objA'): [0.6, 100, 20],
        (subgoal_node, 'objB'): [0.2, 10.222, 34]
    }
    distances = {(robot_node, subgoal_node): 23}

    action = Action(subgoal_node, (target_object,), subgoal_prop_dict)
    robot.retarget(action, distances)

    assert robot.time_remaining >= distances[(robot_node, subgoal_node)]
    assert robot.info_time >= distances[(robot_node, subgoal_node)]

    # Without any history, we encounter a chance node at the 'info time'
    history_none = History()
    event_outcome, event_time = get_next_event_and_time(robot, history_none)
    assert event_outcome == EventOutcome.CHANCE
    assert event_time == distances[(robot_node, subgoal_node)] + \
        min(subgoal_prop_dict[(subgoal_node, target_object)][1],
            subgoal_prop_dict[(subgoal_node, target_object)][2])

    # With a success history, we succeed and have to travel all of RS
    history_succ = History()
    history_succ.add_event(action, EventOutcome.SUCCESS)
    event_outcome, event_time = get_next_event_and_time(robot, history_succ)
    assert event_outcome == EventOutcome.SUCCESS
    assert event_time == distances[(robot_node, subgoal_node)] + \
        subgoal_prop_dict[(subgoal_node, target_object)][1]

    # With a failure history, we fail and have to travel min(RS, RE)
    history_fail = History()
    history_fail.add_event(action, EventOutcome.FAILURE)
    event_outcome, event_time = get_next_event_and_time(robot, history_fail)
    assert event_outcome == EventOutcome.FAILURE
    assert event_time == distances[(robot_node, subgoal_node)] + \
        min(subgoal_prop_dict[(subgoal_node, target_object)][1],
            subgoal_prop_dict[(subgoal_node, target_object)][2])
