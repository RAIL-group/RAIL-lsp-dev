import copy
from enum import Enum

EventOutcome = Enum('EventOutcome', ['CHANCE', 'SUCCESS', 'FAILURE'])


class Node(object):
    def __init__(self, props=[], is_subgoal=False, location=None, frontier=None):
        self.props = props
        self.is_subgoal = is_subgoal
        self.location = location
        self.frontier = frontier
        self.hash_id = hash(str(self.location) + str(self.props) + str(self.is_subgoal))

    def __repr__(self):
        return f'{self.location}'

    def __hash__(self):
        return self.hash_id


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
        self.hash_id = hash(self.target_node) + hash(str(self.props))

    def __eq__(self, other):
        return self.hash_id == other.hash_id

    def __hash__(self):
        return self.hash_id


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

def get_state_from_history(outcome_mrstates, history):
    for state in outcome_mrstates.keys():
        if state.history == history:
            return state

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

class MRState(object):
    def __init__(self, robots, planner, distances,
                 known_space_nodes=[], unknown_space_nodes=[],
                 subgoal_prop_dict = {}, history=History()):
        self.dfa_state = planner.state
        self.robots = robots
        self.distances = distances
        self.planner = planner
        self.known_space_nodes = known_space_nodes
        self.unknown_space_nodes = unknown_space_nodes
        self.subgoal_prop_dict = subgoal_prop_dict
        self.history = history
        if planner.is_accepting_state(self.dfa_state):
            self.is_goal_state = True
        else:
            self.is_goal_state = False

    def transition(self, action):
        temp_state = self.copy(robots=[robot.copy() for robot in self.robots],
                               planner=copy.copy(self.planner),
                               history=self.history.copy())
        needs_action = [robot.needs_action for robot in temp_state.robots]
        if any(needs_action):
            temp_state.robots[needs_action.index(True)].retarget(action, temp_state.distances)
        outcome = advance_mrstate(temp_state)
        # normalize the probabilities (just to be sure)
        total_prob = sum([values[0] for values in outcome.values()])
        normalized_outcome = {key: (values[0]/total_prob, values[1]) for key, values in outcome.items()}
        return normalized_outcome

    def get_actions(self):
        useful_props = self.planner.get_useful_props()
        ks_actions = [Action(node) for node in self.known_space_nodes]
        unk_actions = [Action(node, (props,), self.subgoal_prop_dict)
                       for node in self.unknown_space_nodes
                       for props in useful_props]
        unk_actions = [action for action in unk_actions
                       if self.history.get_action_outcome(action) == EventOutcome.CHANCE]
        return ks_actions + unk_actions

    def copy(self, robots, planner, history):
        return MRState(robots=robots,
                       planner=planner,
                       history=history,
                       distances=self.distances,
                       known_space_nodes=self.known_space_nodes,
                       unknown_space_nodes=self.unknown_space_nodes,
                       subgoal_prop_dict=self.subgoal_prop_dict)

def advance_mrstate(mrstate, prob=1.0, cost=0.0):
    # TODO If any of the robots need an action, return:
    if any(robot.needs_action for robot in mrstate.robots):
        print("robots need actions")
        robots = [robot.copy() for robot in mrstate.robots]
        planner = copy.copy(mrstate.planner)
        history = mrstate.history.copy()
        return {mrstate.copy(robots, planner, history): (prob, cost)}

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
        dfa_planner = copy.copy(mrstate.planner)
        dfa_planner.advance(event_robot.action.props)
        event_robot.reset_needs_action()

        # INFO: if DFA state is updated, check for 'waiting' agents
        # see if they can advance the state as well.
        do_loop = True
        while do_loop:
            do_loop = False
            for robot in robots:
                # [REVISIT: If info time is <=0 (which is set to 0 in the advance function), the robot
                # is waiting.]
                if (robot != event_robot and dfa_planner.does_transition_state(robot.action.props)
                   and robot.info_time == 0):
                    dfa_planner.advance(robot.action.props)
                    robot.reset_needs_action()
                    do_loop = True

        # INFO: if the DFA state is updated, some agents' actions
        # will not be 'useful' anymore and also need retargeting.
        useful_props = dfa_planner.get_useful_props()
        for robot in robots:
            action_prop_still_useful = any([prop in useful_props for prop in robot.action.props])
            if (robot != event_robot and not action_prop_still_useful):
                robot.reset_needs_action()
        state = mrstate.copy(robots=robots, planner=dfa_planner, history=mrstate.history.copy())
        return {state: (prob, cost + event_time)}

    if event_outcome == EventOutcome.CHANCE:
        [robot.advance_time(event_time) for robot in robots]
        # SUCCESS
        success_history = mrstate.history.copy()
        success_history.add_event(event_robot.action, EventOutcome.SUCCESS)

        success_mrstate = mrstate.copy(robots=robots,
                                       planner=copy.copy(mrstate.planner),
                                       history=success_history)
        outcome_states_success = advance_mrstate(success_mrstate,
                                                 cost=cost + event_time,
                                                 prob=prob * event_robot.action.PS)
        # FAILURE
        failure_robots = [robot.copy() for robot in robots]
        event_failure_robot = next(robot for robot in failure_robots if robot == event_robot)
        # find other robots which were pursuing the same action
        event_failure_robot.reset_needs_action()
        other_robots = [robot for robot in failure_robots if robot.action == event_failure_robot.action]
        for robot in other_robots + [event_failure_robot]:
            robot.reset_needs_action()
        failure_history = mrstate.history.copy()
        failure_history.add_event(event_robot.action, EventOutcome.FAILURE)
        failure_mrstate = mrstate.copy(robots=failure_robots,
                                       planner=copy.copy(mrstate.planner),
                                       history=failure_history)
        outcome_state_failure = {
            failure_mrstate: (prob * (1 - event_robot.action.PS), cost + event_time)
        }
        assert (success_history.get_action_outcome(event_robot.action) !=
                failure_history.get_action_outcome(event_robot.action))

        return {**outcome_states_success, **outcome_state_failure}
