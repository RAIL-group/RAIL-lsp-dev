import copy
from mr_task.core import History, Node, Action, EventOutcome
from mr_task.robot import Robot


def test_history_comparison():
    history1 = History()
    history2 = History()
    assert history1 == history2

    subgoal_nodeS1 = Node(is_subgoal=True)
    subgoal_nodeS2 = Node(is_subgoal=True)
    subgoal_prop_dict = {
        (subgoal_nodeS1, 'objA'): [1, 0, 0],
        (subgoal_nodeS1, 'objB'): [1, 0, 0],
        (subgoal_nodeS2, 'objB'): [1, 0, 0],
        (subgoal_nodeS2, 'objA'): [1, 0, 0],
    }
    actionS1A = Action(subgoal_nodeS1, ('objA',), subgoal_prop_dict)
    actionS1B = Action(subgoal_nodeS1, ('objB',), subgoal_prop_dict)
    actionS2A = Action(subgoal_nodeS2, ('objA',), subgoal_prop_dict)
    actionS2B = Action(subgoal_nodeS2, ('objB',), subgoal_prop_dict)

    history1.add_event(actionS1A, EventOutcome.SUCCESS)
    history1.add_event(actionS1B, EventOutcome.FAILURE)

    history2.add_event(actionS1B, EventOutcome.FAILURE)
    history2.add_event(actionS1A, EventOutcome.SUCCESS)

    assert history1 == history2  # Action history added in different order

    history1.add_event(actionS2A, EventOutcome.SUCCESS)
    assert history1 != history2  # one action history less

    history2.add_event(actionS2A, EventOutcome.SUCCESS)
    assert history1 == history2  # Add that action history

    history1.add_event(actionS2B, EventOutcome.SUCCESS)
    history2.add_event(actionS2B, EventOutcome.FAILURE)
    assert history1 != history2  # Different outcome for same action in different history


def test_action_comparison():
    subgoal_nodeS1 = Node(is_subgoal=True, location=(1, 1))
    subgoal_nodeS1_copy = copy.copy(subgoal_nodeS1)
    subgoal_prop_dict = {
        (subgoal_nodeS1, 'objA'): [1, 0, 0],
        (subgoal_nodeS1_copy, 'objA'): [1, 0, 0],
    }
    actionS1A = Action(subgoal_nodeS1, ('objA',), subgoal_prop_dict)
    actionS1A_C = Action(subgoal_nodeS1_copy, ('objA',), subgoal_prop_dict)
    actionS1A_copy = copy.copy(actionS1A)

    assert actionS1A == actionS1A_C
    assert actionS1A == actionS1A_copy


def test_robot_comparison():
    robot_node = Node()
    robot1 = Robot(robot_node)
    robot2 = Robot(robot_node)

    assert robot1 != robot2
    assert hash(robot1) != hash(robot2)
