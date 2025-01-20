import numpy as np
import pytest
from mr_task import DFAManager
from mr_task.core import (MRState,
                          Node,
                          Action,
                          History,
                          EventOutcome,
                          advance_mrstate,
                          get_state_from_history)
from mr_task.robot import Robot


def test_mrstate_transition_function_single_joint_action():
    '''Checks the state transition function for 'single' joint-action'''
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
                      planner=planner,
                      distances=distances)

    outcome_states = mrstate.transition(action_unknownAS1)
    assert len(outcome_states) == 1
    prob, cost = list(outcome_states.values())[0]
    assert cost == 0
    assert prob == 1.0
    mrstate = list(outcome_states.keys())[0]
    outcome_states = mrstate.transition(action_unknownBS2)
    assert len(outcome_states) == 3
    state1_history = History()
    state2_history = History()
    state3_history = History()
    state1_history.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    state1_history.add_event(action_unknownBS2, EventOutcome.SUCCESS)
    state2_history.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    state2_history.add_event(action_unknownBS2, EventOutcome.FAILURE)
    state3_history.add_event(action_unknownAS1, EventOutcome.FAILURE)
    state_history_props = [(state1_history, (0.8 * 0.4, 80)),
                           (state2_history, (0.8 * 0.6, 60)),
                           (state3_history, (0.2, 40))]

    for history in state_history_props:
        state = get_state_from_history(outcome_states, history[0])
        assert outcome_states[state][1] == history[1][1]
        assert pytest.approx(outcome_states[state][0]) == history[1][0]


def test_mrstate_transition_function_reassigning_action_when_robot_finishes():
    '''Checks the state transition function by re-assigning action to the robot which finishes'''
    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True, location='a')
    subgoal_node2 = Node(is_subgoal=True, location='b')
    subgoal_node3 = Node(is_subgoal=True, location='c')

    subgoal_prop_dict = {
        (subgoal_node1, 'objA'): [0.8, 10, 20],
        (subgoal_node2, 'objB'): [0.2, 50, 30],
        (subgoal_node3, 'objB'): [0.5, 110, 100]
    }
    distances = {
        (robot_node, subgoal_node1): 20,
        (robot_node, subgoal_node2): 30,
        (subgoal_node1, subgoal_node3): 190,
        (subgoal_node2, subgoal_node3): 150
    }
    action_unknownAS1 = Action(subgoal_node1, ('objA',), subgoal_prop_dict)
    action_unknownBS2 = Action(subgoal_node2, ('objB',), subgoal_prop_dict)
    action_unknownBS3 = Action(subgoal_node3, ('objB',), subgoal_prop_dict)

    specification = "F objA & F objB"
    planner = DFAManager(specification)
    mrstate = MRState(robots=[robot_unk1, robot_unk2],
                      planner=planner,
                      distances=distances)

    outcome_states = mrstate.transition(action_unknownAS1)
    assert len(outcome_states) == 1
    prob, cost = list(outcome_states.values())[0]
    assert cost == 0
    assert prob == 1.0
    mrstate = list(outcome_states.keys())[0]
    outcome_states = mrstate.transition(action_unknownBS2)
    assert len(outcome_states) == 2

    # Object A is found
    success_AS1_history = History()
    success_AS1_history.add_event(action_unknownAS1, EventOutcome.SUCCESS)
    success_AS1_state = get_state_from_history(outcome_states, success_AS1_history)
    assert success_AS1_state.history.get_action_outcome(action_unknownAS1) == EventOutcome.SUCCESS
    assert success_AS1_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.CHANCE
    assert outcome_states[success_AS1_state][0] == pytest.approx(0.8)
    assert outcome_states[success_AS1_state][1] == 30
    assert not planner.is_accepting_state(success_AS1_state.dfa_state)

    # Object A is NOT found
    failure_AS1_history = History()
    failure_AS1_history.add_event(action_unknownAS1, EventOutcome.FAILURE)
    failure_AS1_state = get_state_from_history(outcome_states, failure_AS1_history)
    assert failure_AS1_state.history.get_action_outcome(action_unknownAS1) == EventOutcome.FAILURE
    assert failure_AS1_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.CHANCE
    assert outcome_states[failure_AS1_state][0] == pytest.approx(0.2)
    assert outcome_states[failure_AS1_state][1] == 30.0
    assert not planner.is_accepting_state(failure_AS1_state.dfa_state)

    # Enumerating: Object A is found
    outcome_success_state = success_AS1_state.transition(action_unknownBS3)
    assert len(outcome_success_state) == 2

    # Object A is found, Object B is found
    success_AS1_success_BS2_history = success_AS1_history.copy()
    success_AS1_success_BS2_history.add_event(action_unknownBS2, EventOutcome.SUCCESS)
    success_AS1_success_BS2_state = get_state_from_history(outcome_success_state, success_AS1_success_BS2_history)
    assert success_AS1_success_BS2_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.SUCCESS
    assert outcome_success_state[success_AS1_success_BS2_state][0] == pytest.approx(0.2)
    assert outcome_success_state[success_AS1_success_BS2_state][1] == 50
    assert planner.is_accepting_state(success_AS1_success_BS2_state.dfa_state)

    # Object A is found, Object B is NOT found
    success_AS1_failure_BS2_history = success_AS1_history.copy()
    success_AS1_failure_BS2_history.add_event(action_unknownBS2, EventOutcome.FAILURE)
    success_AS1_failure_BS2_state = get_state_from_history(outcome_success_state, success_AS1_failure_BS2_history)
    assert success_AS1_failure_BS2_state.history.get_action_outcome(action_unknownBS2) == EventOutcome.FAILURE
    assert outcome_success_state[success_AS1_failure_BS2_state][0] == pytest.approx(1 - 0.2)
    assert outcome_success_state[success_AS1_failure_BS2_state][1] == 30
    assert not planner.is_accepting_state(success_AS1_failure_BS2_state.dfa_state)

    # Enumerating: Object A is NOT Found
    outcome_failure_state = failure_AS1_state.transition(action_unknownBS3)
    assert len(outcome_failure_state) == 2

    # Object A is NOT found, Object B is not found
    failure_AS1_failure_BS2_history = failure_AS1_history.copy()
    failure_AS1_failure_BS2_history.add_event(action_unknownBS2, EventOutcome.FAILURE)
    failure_AS1_failure_BS2_state = get_state_from_history(outcome_failure_state, failure_AS1_failure_BS2_history)
    assert outcome_failure_state[failure_AS1_failure_BS2_state][0] == pytest.approx((1 - 0.2))
    assert outcome_failure_state[failure_AS1_failure_BS2_state][1] == 30
    assert not planner.is_accepting_state(failure_AS1_failure_BS2_state.dfa_state)

    # Object A is NOT found, Object B is not found, Final action assignment (B is not found)
    outcome_failureA_failureB_state = failure_AS1_failure_BS2_state.transition(action_unknownBS3)
    assert len(outcome_failure_state) == 2

    failure_AS1_failure_BS2_failure_BS3_history = failure_AS1_failure_BS2_history.copy()
    failure_AS1_failure_BS2_failure_BS3_history.add_event(action_unknownBS3, EventOutcome.FAILURE)
    failure_AS1_failure_BS2_failure_BS3_state = get_state_from_history(outcome_failureA_failureB_state,
                                                                       failure_AS1_failure_BS2_failure_BS3_history)
    assert outcome_failureA_failureB_state[failure_AS1_failure_BS2_failure_BS3_state][0] == pytest.approx(1 - 0.5)
    assert outcome_failureA_failureB_state[failure_AS1_failure_BS2_failure_BS3_state][1] == 270
    assert not planner.is_accepting_state(failure_AS1_failure_BS2_failure_BS3_state.dfa_state)


def test_mrstate_transition_specification_complete_known_space():
    '''Tests mrstate transition when the specification can be completed by single action in known space'''
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
                      planner=planner,
                      distances=distances)
    child_mrstates = advance_mrstate(mrstate)

    # In this case, the known space node will be reached first
    # Thus the only "child" state will be the success state
    old_dfa_state = mrstate.dfa_state
    assert len(child_mrstates) == 1
    child_mrstate = list(child_mrstates.keys())[0]
    assert not old_dfa_state == child_mrstate.dfa_state
    assert planner.is_accepting_state(child_mrstate.dfa_state)
    assert len([robot for robot in child_mrstate.robots if robot.needs_action]) == 3


def test_mrstate_prop_found_single_robot_remaining_info_time():
    '''Test remaining and info time after reassigning action to the robot which finishes'''
    # Idea: Two robots are assigned to 2 subgoal nodes: one A and one B. One of those finishes and gets reassigned

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
                      planner=planner,
                      distances=distances)
    outcome_mrstates = advance_mrstate(mrstate)

    # Two outcome_mrstates (either objA is found or not found)
    assert len(outcome_mrstates) == 2
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
            min(subgoal_prop_dict[(subgoal_node3, 'objB')][1],
                subgoal_prop_dict[(subgoal_node3, 'objB')][2])


def test_mrstate_prop_found_multi_robot_remaining_info_time():
    '''Test remaining and info time after reassigning action to all the robot because prop is found'''
    # Idea: Two robots are assigned to the same subgoal for same object.
    # If one finds the object, both robots are reassigned
    # If one does not, only that robot is reassigned

    robot_node = Node()
    robot_unk1 = Robot(robot_node)
    robot_unk2 = Robot(robot_node)

    subgoal_node1 = Node(is_subgoal=True, location='a')
    subgoal_node2 = Node(is_subgoal=True, location='b')
    subgoal_node3 = Node(is_subgoal=True, location='c')

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
                      planner=planner,
                      distances=distances)
    outcome_mrstates = advance_mrstate(mrstate)

    success_history = History()
    success_history.add_event(action_unknownS1A, EventOutcome.SUCCESS)
    success_state = get_state_from_history(outcome_mrstates, success_history)

    failure_history = History()
    failure_history.add_event(action_unknownS1A, EventOutcome.FAILURE)
    failure_state = get_state_from_history(outcome_mrstates, failure_history)
    # Two outcome_mrstates (either objA is found or not found)
    # In SUCCESS STATE, BOTH R1 and R2 needs to be re-targeted
    assert success_state.history.get_action_outcome(action_unknownS1A)==EventOutcome.SUCCESS
    assert len([robot for robot in success_state.robots if robot.needs_action]) == 2

    # In FAILURE STATE, ONLY R1 needs to be re-targeted
    assert len([robot for robot in failure_state.robots if robot.needs_action]) == 1
    assert failure_state.history.get_action_outcome(action_unknownS1A)==EventOutcome.FAILURE

    success_state.robots[0].retarget(action_unknownS3B, distances)
    success_state.robots[1].retarget(action_unknownS3C, distances)
    assert success_state.robots[0].time_remaining == 250  # info time is the same
    assert success_state.robots[0].info_time == 250
    assert success_state.robots[1].time_remaining == 280  # info time is 270
    assert success_state.robots[1].info_time == 270

    failure_state.robots[0].retarget(action_unknownS3B, distances)
    assert failure_state.robots[0].time_remaining == 250
    assert failure_state.robots[0].info_time == 250


def test_mrstate_no_actions_with_action_outcome_in_history():
    '''The state should not have the action whose outcome is already in the history'''
    num_robots = 2
    num_subgoals = 2
    robot_node = [Node() for _ in range(num_robots)]
    robots = [Robot(node) for node in robot_node]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    subgoal_prop_dict = {(subgoal, 'goal'): [1, 0, 0] for subgoal in subgoal_nodes}
    distances = {}
    specification = "F goal"
    history = History()
    actions = [
        Action(subgoal, ('goal',), subgoal_prop_dict) for subgoal in subgoal_nodes
    ]
    for action in actions:
        history.add_event(action, np.random.choice([EventOutcome.SUCCESS,
                                                    EventOutcome.FAILURE]))

    planner = DFAManager(specification)
    mrstate = MRState(robots=robots, planner=planner, distances=distances,
                      subgoal_prop_dict=subgoal_prop_dict,
                      unknown_space_nodes=subgoal_nodes,
                      history=history)
    assert len(mrstate.get_actions()) == 0
