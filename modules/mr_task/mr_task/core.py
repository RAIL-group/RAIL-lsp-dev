import copy
from enum import Enum

EventOutcome = Enum('EventOutcome', ['CHANCE', 'SUCCESS', 'FAILURE'])

class Node(object):
    def __init__(self, props=[], is_subgoal=False):
        self.props = props
        self.is_subgoal = is_subgoal


class Action(object):
    def __init__(self, target_node, props=None, subgoal_prop_dict=None):
        if target_node.is_subgoal and subgoal_prop_dict is None:
            raise ValueError('If target_node is a subgoal node, '
                             'the subgoal prop dictionary is required.')
        elif target_node.is_subgoal and (props is None or not len(props) == 1):
            raise ValueError('Only a single prop is allowed for a subgoal node.')
        elif not target_node.is_subgoal and props is not None:
            raise ValueError('"props" should not be provided for a known space node.')

        self.target_node = target_node

        if target_node.is_subgoal:
            self.props = props
            self.PS, self.RS, self.RE = subgoal_prop_dict[(target_node, props[0])]
        else:
            self.props = target_node.props
            self.PS = 1.0
            self.RS = 0.0
            self.RE = 0.0


class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_event(self, action, outcome):
        assert outcome == EventOutcome.SUCCESS or outcome == EventOutcome.FAILURE
        self._data[(action.target_node, action.props[0])] = outcome

    def get_action_outcome(self, action):
        # Return the history or, if it doesn't exist, CHANCE
        return self._data.get((action.target_node, action.props[0]),
                              EventOutcome.CHANCE)

    def copy(self):
        return History(data=self._data.copy())

    def __str__(self):
        return f'{self._data}'

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return self._data == other._data


class MRState(object):
    def __init__(self, robots, planner, history=None, old_state=None, cost=0.0, prob=1.0):
        self.dfa_state = planner.state
        self.robots = robots
        self.planner = planner
        if history is None:
            self.history = History()
        else:
            self.history = history
        if old_state is None:
            self.old_state = old_state
            self.cost = cost
            self.prob = 1.0
        else:
            self.old_state = old_state
            self.cost = old_state.cost + cost
            self.prob = old_state.prob * prob

    def get_outcome_states(self, action, distances):
        needs_action = [robot.needs_action for robot in self.robots]
        if any(needs_action):
            idx = needs_action.index(True)
            self.robots[needs_action.index(True)].retarget(action, distances)
        outcome_states = advance_mrstate(self)
        outcome_states_dict = {state.prob: state for state in outcome_states}
        return outcome_states

    def transition(self, action):
        pass

    def get_actions(self):
        pass


def get_state_with_history(outcome_mrstates, history):
    for state in outcome_mrstates:
        if state.history == history:
            return state

def get_state_from_history_new(outcome_mrstates, history):
    for state in outcome_mrstates.keys():
        if state.history == history:
            return state

def flatten(states):
    flat_list = []
    for state in states:
        if isinstance(state, list):
            flat_list.extend(flatten(state))
        else:
            flat_list.append(state)
    return flat_list

def get_next_event_and_time(robot, history):
    # If the node isn't a subgoal, return SUCCESS and the remaining time
    if not robot.action.target_node.is_subgoal:
        return EventOutcome.SUCCESS, robot.time_remaining

    # If it's a subgoal, we need to figure out if the history includes
    # this particular outcome
    outcome = history.get_action_outcome(robot.action)
    if outcome == EventOutcome.SUCCESS:
        return outcome, robot.time_remaining
    else:
        return outcome, robot.info_time

def advance_mrstate(mrstate):
    # TODO If any of the robots need an action, return:
    if any(robot.needs_action for robot in mrstate.robots):
        print("robots need actions")
        robots = [robot.copy() for robot in mrstate.robots]
        return [MRState(robots, copy.copy(mrstate.planner), copy.copy(mrstate.history), old_state=mrstate)]

    # Propagate the MRState class as far as it will go
    robots = [robot.copy() for robot in mrstate.robots]
    events = [(robot, *get_next_event_and_time(robot, mrstate.history))
              for robot in robots
              if mrstate.planner.does_transition_state(robot.action.props)]
    shortest_event = min(events, key=lambda x: x[2])
    event_robot, event_outcome, event_time = shortest_event

    if event_outcome == EventOutcome.SUCCESS:
        # If SUCCESS, advance time and return
        # History unchanged, so copy unnecessary
        [robot.advance_time(event_time) for robot in robots]
        mrstate.planner.advance(event_robot.action.props)
        event_robot.reset_needs_action()

        # INFO: if DFA state is updated, check for 'waiting' agents
        # see if they can advance the state as well.
        do_loop = True
        while do_loop:
            do_loop = False
            for robot in robots:
                # [REVISIT: If info time is <=0 (which is set to 0 in the advance function), the robot
                # is waiting.]
                if (robot != event_robot and mrstate.planner.does_transition_state(robot.action.props)
                   and robot.info_time == 0):
                    mrstate.planner.advance(robot.action.props)
                    robot.reset_needs_action()
                    do_loop = True

        # INFO: if the DFA state is updated, some agents' actions
        # will not be 'useful' anymore and also need retargeting.
        useful_props = mrstate.planner.get_useful_props()
        for robot in robots:
            action_prop_still_useful = any([prop in useful_props for prop in robot.action.props])
            if (robot != event_robot and not action_prop_still_useful):
                robot.reset_needs_action()

        return [MRState(robots, copy.copy(mrstate.planner), copy.copy(mrstate.history),
                        old_state=mrstate, cost=event_time, prob=1.0)]

    if event_outcome == EventOutcome.CHANCE:
        [robot.advance_time(event_time) for robot in robots]
        # SUCCESS
        success_history = mrstate.history.copy()
        success_history.add_event(event_robot.action, EventOutcome.SUCCESS)

        success_mrstate = MRState(robots, copy.copy(mrstate.planner), success_history,
                                old_state=mrstate, cost=event_time, prob=event_robot.action.PS)

        # FAILURE
        failure_robots = [robot.copy() for robot in robots]
        event_failure_robot = next(robot for robot in failure_robots if robot == event_robot)
        event_failure_robot.reset_needs_action()
        failure_history = mrstate.history.copy()
        failure_history.add_event(event_robot.action, EventOutcome.FAILURE)

        failure_mrstate = MRState(failure_robots, copy.copy(mrstate.planner), failure_history,
                                  old_state=mrstate, cost=event_time, prob=(1 - event_robot.action.PS))

        assert (success_history.get_action_outcome(event_robot.action) !=
                failure_history.get_action_outcome(event_robot.action))

        return flatten([advance_mrstate(success_mrstate), failure_mrstate])

class MRStateNew(object):
    def __init__(self, robots, planner, distances, history=None):
        self.dfa_state = planner.state
        self.robots = robots
        self.distances = distances
        self.planner = planner
        if history is None:
            self.history = History()
        else:
            self.history = history

    def transition(self, action):
        needs_action = [robot.needs_action for robot in self.robots]
        if any(needs_action):
            self.robots[needs_action.index(True)].retarget(action, self.distances)
        outcome = advance_mrstate_new(self)
        return outcome

    def get_actions(self):
        pass

def advance_mrstate_new(mrstate, cost=0, prob=1.0):
    # TODO If any of the robots need an action, return:
    if any(robot.needs_action for robot in mrstate.robots):
        print("robots need actions")
        robots = [robot.copy() for robot in mrstate.robots]
        state = MRStateNew(robots=robots, planner=copy.copy(mrstate.planner),
                           history=copy.copy(mrstate.history), distances=mrstate.distances)
        return {state: (cost, prob)}

    # Propagate the MRState class as far as it will go
    robots = [robot.copy() for robot in mrstate.robots]
    events = [(robot, *get_next_event_and_time(robot, mrstate.history))
              for robot in robots
              if mrstate.planner.does_transition_state(robot.action.props)]
    shortest_event = min(events, key=lambda x: x[2])
    event_robot, event_outcome, event_time = shortest_event

    if event_outcome == EventOutcome.SUCCESS:
        # If SUCCESS, advance time and return
        # History unchanged, so copy unnecessary
        [robot.advance_time(event_time) for robot in robots]
        mrstate.planner.advance(event_robot.action.props)
        event_robot.reset_needs_action()

        # INFO: if DFA state is updated, check for 'waiting' agents
        # see if they can advance the state as well.
        do_loop = True
        while do_loop:
            do_loop = False
            for robot in robots:
                # [REVISIT: If info time is <=0 (which is set to 0 in the advance function), the robot
                # is waiting.]
                if (robot != event_robot and mrstate.planner.does_transition_state(robot.action.props)
                   and robot.info_time == 0):
                    mrstate.planner.advance(robot.action.props)
                    robot.reset_needs_action()
                    do_loop = True

        # INFO: if the DFA state is updated, some agents' actions
        # will not be 'useful' anymore and also need retargeting.
        useful_props = mrstate.planner.get_useful_props()
        for robot in robots:
            action_prop_still_useful = any([prop in useful_props for prop in robot.action.props])
            if (robot != event_robot and not action_prop_still_useful):
                robot.reset_needs_action()
        state = MRStateNew(robots=robots, planner=copy.copy(mrstate.planner),
                            history=copy.copy(mrstate.history), distances=mrstate.distances)
        return {state: (cost + event_time, prob)}

    if event_outcome == EventOutcome.CHANCE:
        [robot.advance_time(event_time) for robot in robots]
        # SUCCESS
        success_history = mrstate.history.copy()
        success_history.add_event(event_robot.action, EventOutcome.SUCCESS)

        success_mrstate = MRStateNew(robots=robots,
                                     planner=copy.copy(mrstate.planner),
                                     history=success_history,
                                     distances=mrstate.distances)
        outcome_states_success = advance_mrstate_new(success_mrstate,
                                                     cost=cost + event_time,
                                                     prob=prob * event_robot.action.PS)
        # FAILURE
        failure_robots = [robot.copy() for robot in robots]
        event_failure_robot = next(robot for robot in failure_robots if robot == event_robot)
        event_failure_robot.reset_needs_action()
        failure_history = mrstate.history.copy()
        failure_history.add_event(event_robot.action, EventOutcome.FAILURE)
        failure_mrstate = MRStateNew(robots=failure_robots,
                                     planner=copy.copy(mrstate.planner),
                                     history=failure_history,
                                     distances=mrstate.distances)
        outcome_state_failure = {failure_mrstate: (cost + event_time, prob * (1 - event_robot.action.PS))}
        assert (success_history.get_action_outcome(event_robot.action) !=
                failure_history.get_action_outcome(event_robot.action))

        return {**outcome_states_success, **outcome_state_failure}
