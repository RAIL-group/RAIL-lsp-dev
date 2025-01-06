import pytest
from mr_task import DFAManager
from mr_task.core import (MRStateNew,
                          Node,
                          Action,
                          History,
                          EventOutcome,
                          get_state_with_history,
                          advance_mrstate,
                          get_state_from_history_new)
from mr_task.robot import Robot

def test_state_transition_props():
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
    mrstate = MRStateNew(robots=[robot_unk1, robot_unk2],
                      planner=planner,
                      distances=distances)

    outcome_states = mrstate.transition(action_unknownAS1)
    assert len(outcome_states) == 1
    cost, prob = list(outcome_states.values())[0]
    assert cost == 0
    assert prob == 1.0

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
    state_history_props = [(state1_history, (80, 0.8 * 0.4)),
                          (state2_history, (60, 0.8 * 0.6)),
                          (state3_history, (40, 0.2))]

    for history in state_history_props:
        state = get_state_from_history_new(outcome_states, history[0])
        assert outcome_states[state][0] == history[1][0]
        assert pytest.approx(outcome_states[state][1]) == history[1][1]
