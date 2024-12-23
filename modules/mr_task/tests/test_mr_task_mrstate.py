import pytest
from mr_task import DFAManager
from mr_task.core import (MRState,
                          Node,
                          Action,
                          History,
                          EventOutcome,
                          get_state_with_history,
                          advance_mrstate)
from mr_task.robot import Robot


def test_mrtask_mrstate_outcome_states_test_with_history():
    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True)
    subgoal_node2 = Node(is_subgoal=True)
    subgoal_node3 = Node(is_subgoal=True)

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [0.8, 10, 20],
        (subgoal_node2, 'objB'): [0.2, 50, 30],
        (subgoal_node3, 'objB'): [0.5, 110, 100]
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 30,
        (subgoal_node1, subgoal_node3): 200,
        (subgoal_node2, subgoal_node3): 150
    }
    action_unknownAS1 = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownBS2 = Action(subgoal_node2, ('objB',), subgoal_prop_dict)
    action_unknownBS3 = Action(subgoal_node3, ('objB',), subgoal_prop_dict)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner)
    print(subgoal_node1, subgoal_node2, subgoal_node3)

    outcome_states = mrstate.get_outcome_states(action_unknownAS1, distances)
    assert len(outcome_states) == 1
    outcome_states = outcome_states[0].get_outcome_states(action_unknownBS2, distances)
    assert len(outcome_states) == 2

    success_AS1_history = History()
    success_AS1_history.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    success_AS1_state = get_state_with_history(outcome_states, success_AS1_history)
    assert success_AS1_state.history.get_action_outcome(action_unknownAS1) == EventOutcome.SUCCESS
    assert success_AS1_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.CHANCE
    assert success_AS1_state.cost == 30
    assert success_AS1_state.prob == pytest.approx(0.8)
    assert not planner.is_accepting_state(success_AS1_state.dfa_state)

    failure_AS1_history = History()
    failure_AS1_history.add_event(action_unknownAS1, EventOutcome.FAILURE)
    failure_AS1_state = get_state_with_history(outcome_states, failure_AS1_history)
    assert failure_AS1_state.history.get_action_outcome(action_unknownAS1) == EventOutcome.FAILURE
    assert failure_AS1_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.CHANCE
    assert failure_AS1_state.cost == 30.0
    assert failure_AS1_state.prob == pytest.approx(0.2)
    assert not planner.is_accepting_state(failure_AS1_state.dfa_state)

    # Object A is found
    outcome_success_state = success_AS1_state.get_outcome_states(action_unknownBS3, distances)
    assert len(outcome_success_state) == 2

    # Object A is found, Object B is found
    success_AS1_success_BS2_history = success_AS1_history.copy()
    success_AS1_success_BS2_history.add_event(action_unknownBS2, EventOutcome.SUCCESS)
    success_AS1_success_BS2_state = get_state_with_history(outcome_success_state, success_AS1_success_BS2_history)
    assert success_AS1_success_BS2_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.SUCCESS
    assert success_AS1_success_BS2_state.cost == 80
    assert success_AS1_success_BS2_state.prob == pytest.approx(0.8 * 0.2)
    assert planner.is_accepting_state(success_AS1_success_BS2_state.dfa_state)

    # Object A is found, Object B is NOT found
    success_AS1_failure_BS2_history = success_AS1_history.copy()
    success_AS1_failure_BS2_history.add_event(action_unknownBS2, EventOutcome.FAILURE)
    success_AS1_failure_BS2_state = get_state_with_history(outcome_success_state, success_AS1_failure_BS2_history)
    assert success_AS1_failure_BS2_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.FAILURE
    assert success_AS1_failure_BS2_state.cost == 60
    assert success_AS1_failure_BS2_state.prob == pytest.approx(0.8 * (1 - 0.2))
    assert not planner.is_accepting_state(success_AS1_failure_BS2_state.dfa_state)

    # Object A is NOT Found
    outcome_failure_state = failure_AS1_state.get_outcome_states(action_unknownBS3, distances)
    assert len(outcome_failure_state) == 2

    # Object A is NOT found, Object B is not found
    failure_AS1_failure_BS2_history = failure_AS1_history.copy()
    failure_AS1_failure_BS2_history.add_event(action_unknownBS2, EventOutcome.FAILURE)
    failure_AS1_failure_BS2_state = get_state_with_history(outcome_failure_state, failure_AS1_failure_BS2_history)
    assert failure_AS1_failure_BS2_state.cost == 60
    assert failure_AS1_failure_BS2_state.prob == pytest.approx((1 - 0.8) * (1 - 0.2))
    assert not planner.is_accepting_state(failure_AS1_failure_BS2_state.dfa_state)

    # Object A is NOT found, Object B is not found, Final action assignment (B is not found)
    outcome_failureA_failureB_state = failure_AS1_failure_BS2_state.get_outcome_states(action_unknownBS3, distances)
    assert len(outcome_failure_state) == 2

    failure_AS1_failure_BS2_failure_BS3_history = failure_AS1_failure_BS2_history.copy()
    failure_AS1_failure_BS2_failure_BS3_history.add_event(action_unknownBS3, EventOutcome.FAILURE)
    failure_AS1_failure_BS2_failure_BS3_state = get_state_with_history(outcome_failureA_failureB_state,
                                                                       failure_AS1_failure_BS2_failure_BS3_history)
    assert failure_AS1_failure_BS2_failure_BS3_state.cost == 60 + 280
    assert failure_AS1_failure_BS2_failure_BS3_state.prob == pytest.approx((1-0.8) * (1 - 0.2) * (1 - 0.5))
    assert not planner.is_accepting_state(failure_AS1_failure_BS2_failure_BS3_state.dfa_state)

def test_mrtask_more_mrstate_outcomes():
    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True)
    subgoal_node2 = Node(is_subgoal=True)

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [0.8, 100, 20],
        (subgoal_node2, 'objB'): [0.4, 50, 30],
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 30,
    }
    action_unknownAS1 = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownBS2 = Action(subgoal_node2, ('objB',), subgoal_prop_dict)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner)
    outcome_states = [mrstate]
    for action in [action_unknownAS1, action_unknownBS2]:
        outcome_states = outcome_states[0].get_outcome_states(action, distances)
    assert len(outcome_states) == 3

    history_A_F = History()
    history_A_F.add_event(action_unknownAS1, EventOutcome.FAILURE)
    A_F_state = get_state_with_history(outcome_states, history_A_F)
    assert A_F_state.cost == 40
    assert A_F_state.prob == pytest.approx(0.2)
    assert not planner.is_accepting_state(A_F_state.dfa_state)

    history_A_S_B_S = History()
    history_A_S_B_S.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    history_A_S_B_S.add_event(action_unknownBS2, EventOutcome.SUCCESS)
    A_S_B_S_state = get_state_with_history(outcome_states, history_A_S_B_S)
    assert A_S_B_S_state.cost == 80
    assert A_S_B_S_state.prob == pytest.approx(0.8 * 0.4)
    assert planner.is_accepting_state(A_S_B_S_state.dfa_state)

    history_A_S_B_F = History()
    history_A_S_B_F.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    history_A_S_B_F.add_event(action_unknownBS2, EventOutcome.FAILURE)
    A_S_B_F_state = get_state_with_history(outcome_states, history_A_S_B_F)
    assert A_S_B_F_state.cost == 60
    assert A_S_B_F_state.prob == pytest.approx(0.8 * (1- 0.4))
    assert not planner.is_accepting_state(A_S_B_F_state.dfa_state)


def test_mrtask_mrstate_advancement():
    # Set up the environment
    robot_node = Node()
    robot_known = Robot(robot_node)
    robot_unknownA = Robot(robot_node)
    robot_unknownB = Robot(robot_node)
    subgoal_node = Node(is_subgoal=True)
    known_space_node = Node(props=('objA', 'objB',))
    subgoal_prop_dict = {
        (subgoal_node, 'objA'): [0.6, 100, 20],
        (subgoal_node, 'objB'): [0.2, 10.222, 34]
    }
    distances = {
        (robot_node, subgoal_node): 23,
        (robot_node, known_space_node): 1
    }

    action_known = Action(known_space_node)
    action_unknownA = Action(subgoal_node, ('objA',), subgoal_prop_dict)
    action_unknownB = Action(subgoal_node, ('objB',), subgoal_prop_dict)

    robot_known.retarget(action_known, distances)
    robot_unknownA.retarget(action_unknownA, distances)
    robot_unknownB.retarget(action_unknownB, distances)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_known, robot_unknownA, robot_unknownB],
                      planner=planner)
    child_mrstates = advance_mrstate(mrstate)

    # In this case, the known space node will be reached first
    # Thus the only "child" state will be the success state
    old_dfa_state = mrstate.dfa_state
    assert len(child_mrstates) == 1
    assert not old_dfa_state == child_mrstates[0].dfa_state
    assert planner.is_accepting_state(child_mrstates[0].dfa_state)
    assert len([robot for robot in child_mrstates[0].robots if robot.needs_action]) == 3


def test_mrtask_mrstate_advance_act_advance_1():
    # Idea:
    # - Spec A & B & C
    # - Simple environment with 2 subgoal nodes: one A and one B
    # - One of those finishes
    # - Then assign another action to the robot needing reassignment (something *very* far away, objC)
    # -  Confirm the 'time remaining' computed correctly.
    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True)
    subgoal_node2 = Node(is_subgoal=True)
    subgoal_node3 = Node(is_subgoal=True)

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [0.8, 10, 20],
        (subgoal_node2, 'objB'): [0.2, 50, 30],
        (subgoal_node3, 'objB'): [0.5, 110, 100]
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 30,
        (subgoal_node1, subgoal_node3): 200,
        (subgoal_node2, subgoal_node3): 150
    }
    action_unknownA = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownB = Action(subgoal_node2, ('objB',), subgoal_prop_dict)
    action_unknownC = Action(subgoal_node3, ('objB',), subgoal_prop_dict)

    robot_unk1.retarget(action_unknownA, distances)
    robot_unk2.retarget(action_unknownB, distances)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner)
    outcome_mrstates = advance_mrstate(mrstate)

    # Two outcome_mrstates (either objA is found or not found)
    assert len(outcome_mrstates) == 2
    assert outcome_mrstates[0].history.get_action_outcome(action_unknownA) == EventOutcome.SUCCESS
    assert outcome_mrstates[1].history.get_action_outcome(action_unknownA) == EventOutcome.FAILURE
    for outcome_mrstate in outcome_mrstates:
        # In both success and failure states, R1 needs to be re-assigned
        assert len([robot for robot in outcome_mrstate.robots if robot.needs_action]) == 1
        retarget_robot = [robot for robot in outcome_mrstate.robots if robot.needs_action][0]
        assert retarget_robot.time_remaining == 0
        assert retarget_robot.info_time == 0
        other_robot = [robot for robot in outcome_mrstate.robots if not robot.needs_action][0]
        # TODO: Change this with variables
        assert other_robot.time_remaining == 50
        assert other_robot.info_time == 30

        # make the finished robot take a different action
        retarget_robot.retarget(action_unknownC, distances)

        # check the remaining_time for retargeted robot
        # remaining_time = 320, info_time = 310
        assert retarget_robot.time_remaining == subgoal_prop_dict[(subgoal_node1, 'objA')][1] + \
                                                distances[(subgoal_node1, subgoal_node3)] + \
                                                subgoal_prop_dict[(subgoal_node3, 'objB')][1]
        assert retarget_robot.info_time == subgoal_prop_dict[(subgoal_node1, 'objA')][1] + \
                                            distances[(subgoal_node1, subgoal_node3)] + \
                                            min(subgoal_prop_dict[(subgoal_node3, 'objB')][1], subgoal_prop_dict[(subgoal_node3, 'objB')][2])


def test_mrtask_mrstate_advance_act_advance_1_with_history():
    # Idea:
    # - Spec A & B & C
    # - Simple environment with 2 subgoal nodes: one A and one B
    # - One of those finishes
    # - Then assign another action to the robot needing reassignment (something *very* far away, objC)
    # -  Confirm the 'time remaining' computed correctly.
    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True)
    subgoal_node2 = Node(is_subgoal=True)
    subgoal_node3 = Node(is_subgoal=True)

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [0.8, 10, 20],
        (subgoal_node2, 'objB'): [0.2, 50, 30],
        (subgoal_node3, 'objB'): [0.5, 110, 100]
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 30,
        (subgoal_node1, subgoal_node3): 200,
        (subgoal_node2, subgoal_node3): 150
    }
    action_unknownS1A = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownS2B = Action(subgoal_node2, ('objB',), subgoal_prop_dict)
    action_unknownS3B = Action(subgoal_node3, ('objB',), subgoal_prop_dict)

    robot_unk1.retarget(action_unknownS1A, distances)
    robot_unk2.retarget(action_unknownS2B, distances)

    specification = "F objA & F objB"
    # With a success history, we succeed and have to travel all of RS
    history = History()
    history.add_event(action_unknownS1A, EventOutcome.SUCCESS)
    history.add_event(action_unknownS2B, EventOutcome.FAILURE)
    history.add_event(action_unknownS3B, EventOutcome.SUCCESS)
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner, history=history)
    outcomes = advance_mrstate(mrstate)


def test_mrtask_mrstate_advance_act_advance_2():
    # pass
    # Idea:
    # - Spec A & B & C
    # - Simple environment with 2 subgoal nodes: *both* objA
    # - One of those succeeds and as a result, *both* robots start heading to the other node. Confirm the 'time remaining' computed correctly.
    # - Then assign another action to the robot needing reassignment (something *very* far away, objC)
    # - Confirm that it reaches that final target at the 'right time'.

    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True)
    subgoal_node2 = Node(is_subgoal=True)
    subgoal_node3 = Node(is_subgoal=True)

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [1, 10, 20], # make sure this succeeds so that single child child is returned first
        (subgoal_node2, 'objA'): [0.2, 50, 30],
        (subgoal_node3, 'objC'): [0.5, 110, 100],
        (subgoal_node3, 'objB'): [0.5, 40, 80]
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 10,
        (subgoal_node1, subgoal_node3): 200,
        (subgoal_node2, subgoal_node3): 150

    }
    action_unknownS1A = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownS2A = Action(subgoal_node2, ('objA',), subgoal_prop_dict)
    action_unknownS3C = Action(subgoal_node3, ('objC',), subgoal_prop_dict)
    action_unknownS3B = Action(subgoal_node3, ('objB',), subgoal_prop_dict)

    robot_unk1.retarget(action_unknownS1A, distances)
    robot_unk2.retarget(action_unknownS2A, distances)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner)
    outcome_mrstates = advance_mrstate(mrstate)
    assert len(outcome_mrstates) == 2
    success_state, failure_state = outcome_mrstates[0], outcome_mrstates[1]
    # Two outcome_mrstates (either objA is found or not found)
    # In SUCCESS STATE, BOTH R1 and R2 needs to be re-targeted
    assert success_state.history.get_action_outcome(action_unknownS1A)==EventOutcome.SUCCESS
    assert len([robot for robot in success_state.robots if robot.needs_action]) == 2

    # In FAILURE STATE, ONLY R1 needs to be re-targeted
    assert len([robot for robot in failure_state.robots if robot.needs_action]) == 1
    assert failure_state.history.get_action_outcome(action_unknownS1A)==EventOutcome.FAILURE

    success_state.robots[0].retarget(action_unknownS3B, distances)
    success_state.robots[1].retarget(action_unknownS3C, distances)
    assert success_state.robots[0].time_remaining == 250 # info time is the same
    assert success_state.robots[0].info_time == 250
    assert success_state.robots[1].time_remaining == 280 # info time is 270
    assert success_state.robots[1].info_time == 270

    failure_state.robots[0].retarget(action_unknownS3B, distances)
    assert failure_state.robots[0].time_remaining == 250
    assert failure_state.robots[0].info_time == 250
