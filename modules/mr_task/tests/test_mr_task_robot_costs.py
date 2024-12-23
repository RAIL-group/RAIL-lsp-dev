import pytest
from mr_task.core import Node, Action
from mr_task.robot import Robot


@pytest.mark.parametrize('delta_t', [9, 5, 0])
def test_mrtask_robot_action_switch_known(delta_t):
    '''Robot goes to KSN_A. In mid-way, it has to switch action. Now it has to go KSN_B'''
    robot_node = Node()
    robot = Robot(robot_node)
    known_space_nodeA = Node(props=('objA',))
    known_space_nodeB = Node(props=('objB',))
    print(robot_node, known_space_nodeA, known_space_nodeB)
    actionA = Action(known_space_nodeA)
    actionB = Action(known_space_nodeB)
    distances = {(robot_node, known_space_nodeA): 10,
                (robot_node, known_space_nodeB): 15}
    robot.retarget(actionA, distances)
    assert robot.time_remaining == 10
    assert robot.info_time == 10
    assert delta_t <= robot.time_remaining
    robot.advance_time(delta_t)
    robot.reset_needs_action()
    robot.retarget(actionB, distances)
    assert robot.time_remaining == delta_t + distances[(robot_node, actionB.target_node)]

@pytest.mark.parametrize('delta_t', [10, 50, 110])
def test_mrtask_robot_action_switch_unknown(delta_t):
    '''Robot goes to SN_A. In mid-way, it has to switch action. Now it has to go SN_B'''
    robot_node = Node()
    robot = Robot(robot_node)
    subgoal_nodeA = Node(is_subgoal=True)
    subgoal_nodeB = Node(is_subgoal=True)
    subgoal_prop_dict = {
        (subgoal_nodeA, 'objA'): [0.6, 100, 20],
        (subgoal_nodeB, 'objB'): [0.2, 10, 34]
    }
    actionA = Action(subgoal_nodeA, ('objA',), subgoal_prop_dict)
    actionB = Action(subgoal_nodeB, ('objB',), subgoal_prop_dict)

    distances = {(robot_node, subgoal_nodeA): 10,
                (robot_node, subgoal_nodeB): 15,
                (subgoal_nodeA, subgoal_nodeB): 5}
    robot.retarget(actionA, distances)
    assert robot.time_remaining == 110
    assert robot.info_time == 30
    assert delta_t >= distances[(robot_node, actionA.target_node)]
    robot.advance_time(delta_t)
    robot.reset_needs_action()
    robot.retarget(actionB, distances)
    progress_on_subgoalA = delta_t - distances[(robot_node, subgoal_nodeA)]
    assert robot.time_remaining == progress_on_subgoalA + distances[(subgoal_nodeA, subgoal_nodeB)] + 10

def test_mrstate_robot_action_switch_multiple():
    '''Robot goes to SN_A. In mid-way, it has to switch action multiple times. Now it has to go SN_B'''
    robot_node = Node()
    robot = Robot(robot_node)
    subgoal_nodeA = Node(is_subgoal=True)
    subgoal_nodeB = Node(is_subgoal=True)
    subgoal_nodeC = Node(is_subgoal=True)
    subgoal_nodeD = Node(is_subgoal=True)
    subgoal_prop_dict = {
        (subgoal_nodeA, 'objA'): [0.6, 100, 20],
        (subgoal_nodeB, 'objB'): [0.2, 10, 34],
        (subgoal_nodeC, 'objC'): [0.5, 12, 24],
        (subgoal_nodeD, 'objD'): [1.0, 200, 20]
    }
    actionA = Action(subgoal_nodeA, ('objA',), subgoal_prop_dict)
    actionB = Action(subgoal_nodeB, ('objB',), subgoal_prop_dict)
    actionC = Action(subgoal_nodeC, ('objC',), subgoal_prop_dict)
    actionD = Action(subgoal_nodeD, ('objD',), subgoal_prop_dict)

    distances = {(robot_node, subgoal_nodeA): 10,
                (subgoal_nodeA, subgoal_nodeA): 0,
                (subgoal_nodeA, subgoal_nodeB): 5, (subgoal_nodeA, subgoal_nodeC): 10, (subgoal_nodeA, subgoal_nodeD): 15,
                (subgoal_nodeB, subgoal_nodeA): 5, (subgoal_nodeB, subgoal_nodeC): 5, (subgoal_nodeB, subgoal_nodeD): 10,
                (subgoal_nodeC, subgoal_nodeA): 10, (subgoal_nodeC, subgoal_nodeB): 5, (subgoal_nodeC, subgoal_nodeD): 5,
                (subgoal_nodeD, subgoal_nodeA): 15, (subgoal_nodeD, subgoal_nodeB): 10, (subgoal_nodeD, subgoal_nodeC): 5}

    robot.retarget(actionA, distances)
    delta_t = 30
    robot.advance_time(delta_t)
    robot.reset_needs_action()
    robot.retarget(actionB, distances)
    assert robot.time_remaining == 35
    assert robot.info_time == 35

    robot.advance_time(5)
    robot.reset_needs_action()
    robot.retarget(actionC, distances)
    assert robot.time_remaining == 37
    assert robot.info_time == 37

    robot.advance_time(5)
    robot.reset_needs_action()
    robot.retarget(actionD, distances)
    assert robot.time_remaining == 25 + 200
    assert robot.info_time == 25 + 20

def test_mrstate_robot_action_switch_same_subgoal_different_object():
    '''Robot goes to SN_A. In mid-way, it has to switch action multiple times. Now it has to go SN_B'''
    robot_node = Node()
    robot = Robot(robot_node)
    subgoal_nodeA = Node(is_subgoal=True)
    subgoal_nodeB = Node(is_subgoal=True)
    subgoal_prop_dict = {
        (subgoal_nodeA, 'objA'): [0.6, 100, 20],
        (subgoal_nodeB, 'objB'): [0.2, 10, 34],
        (subgoal_nodeA, 'objC'): [0.5, 12, 24],
        (subgoal_nodeA, 'objD'): [1.0, 200, 100]
    }
    actionA = Action(subgoal_nodeA, ('objA',), subgoal_prop_dict)
    actionB = Action(subgoal_nodeB, ('objB',), subgoal_prop_dict)
    actionC = Action(subgoal_nodeA, ('objC',), subgoal_prop_dict)
    actionD = Action(subgoal_nodeA, ('objD',), subgoal_prop_dict)

    distances = {(robot_node, subgoal_nodeA): 10,
                (subgoal_nodeA, subgoal_nodeA): 0, (subgoal_nodeA, subgoal_nodeB): 5}

    robot.retarget(actionA, distances)

    robot.advance_time(28)
    robot.reset_needs_action()
    robot.retarget(actionB, distances)
    assert robot.time_remaining == (18 + 5 + 10)
    assert robot.info_time == (18 + 5 + 10)

    robot.advance_time(5)
    robot.reset_needs_action()
    robot.retarget(actionC, distances)
    assert robot.time_remaining == 0
    assert robot.info_time == 0

    robot.advance_time(0)
    robot.reset_needs_action()
    robot.retarget(actionD, distances)
    assert robot.time_remaining == (200 - 13)
    assert robot.info_time == (100 - 13)

    robot.advance_time(50)
    robot.reset_needs_action()
    robot.retarget(actionB, distances)
    assert robot.time_remaining == (63 + 5 + 10)
    assert robot.info_time == (63 + 5 + 10)
