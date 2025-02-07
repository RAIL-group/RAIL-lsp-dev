import copy
from enum import Enum

EventOutcome = Enum('EventOutcome', ['CHANCE', 'SUCCESS', 'FAILURE'])


class Node(object):
    def __init__(self, props=[], is_subgoal=False, location=None, frontier=None, name=None):
        self.name = name
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
                 subgoal_prop_dict={}, history=History()):
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
        ks_actions = [Action(node) for node in self.known_space_nodes if self.planner.does_transition_state(node.props)]
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
    '''This function propagates the state as far as it goes, until some robot needs re-assignment'''

    # 1.) If any robots need an action, return:
    if any(robot.needs_action for robot in mrstate.robots):
        robots = [robot.copy() for robot in mrstate.robots]
        planner = copy.copy(mrstate.planner)
        history = mrstate.history.copy()
        return {mrstate.copy(robots, planner, history): (prob, cost)}

    # 2.) Find the robot that finishes the action first: event_robot along with the outcome and the time it takes
    robots, event_robot, event_outcome, event_time = _get_robot_that_finishes_first(mrstate)
    # 3.) Advance the time (make the robot move there)
    [robot.advance_time(event_time) for robot in robots]

    if event_outcome == EventOutcome.SUCCESS:
        # 1) If SUCCESS, advance the dfa_state
        dfa_planner = copy.copy(mrstate.planner)
        dfa_planner.advance(event_robot.action.props)
        event_robot.reset_needs_action()

        # 2). Check for 'waiting' agents & see if they can advance the state as well.
        # _advance_dfa_if_robots_waiting(event_robot, robots, dfa_planner)

        # 3). Some robots' actions will not be 'useful' anymore and also need retargeting.
        useful_props = dfa_planner.get_useful_props()
        for robot in robots:
            action_prop_still_useful = any([prop in useful_props for prop in robot.action.props])
            if (robot != event_robot and not action_prop_still_useful):
                robot.reset_needs_action()

        # 4). Return the new state
        state = mrstate.copy(robots=robots, planner=dfa_planner, history=mrstate.history.copy())
        return {state: (prob, cost + event_time)}

    if event_outcome == EventOutcome.CHANCE:
        # 1. Create a SUCCESS state from mrstate with the event_robot's action as SUCCESS
        success_history = mrstate.history.copy()
        success_history.add_event(event_robot.action, EventOutcome.SUCCESS)
        success_mrstate = mrstate.copy(robots=robots,
                                       planner=copy.copy(mrstate.planner),
                                       history=success_history)

        # 2. Propagate the success state again as no robots need re-targeting
        outcome_states_success = advance_mrstate(success_mrstate,
                                                 cost=cost + event_time,
                                                 prob=prob * event_robot.action.PS)

        # 3. Create a FAILURE state from mrstate with the event_robot's action as FAILURE
        failure_history = mrstate.history.copy()
        failure_history.add_event(event_robot.action, EventOutcome.FAILURE)
        # 3.1 Find all the robots those were doing the same action, they need re-targeting
        failure_robots = [robot.copy() for robot in robots]
        event_failure_robots = [robot for robot in failure_robots if robot.action == event_robot.action]
        for robot in event_failure_robots:
            robot.reset_needs_action()
        # 3.2 return the state
        failure_mrstate = mrstate.copy(robots=failure_robots,
                                       planner=copy.copy(mrstate.planner),
                                       history=failure_history)
        outcome_state_failure = {failure_mrstate: (prob * (1 - event_robot.action.PS), cost + event_time)}

        return {**outcome_states_success, **outcome_state_failure}


def _get_robot_that_finishes_first(mrstate):
    all_robots = [robot.copy() for robot in mrstate.robots]
    events = [(robot, *get_next_event_and_time(robot, mrstate.history))
              for robot in all_robots
              if mrstate.planner.does_transition_state(robot.action.props)]
    shortest_event = min(events, key=lambda x: x[2])
    event_robot, event_outcome, event_time = shortest_event
    return all_robots, event_robot, event_outcome, event_time


def _advance_dfa_if_robots_waiting(event_robot, robots, dfa_planner):
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


class RobotNode(object):
    _id_counter = 0

    def __init__(self, start_node):
        self.start = start_node
        self.action = None
        self.needs_action = True
        self._cost_to_target = 0
        self._start_offset = 0
        self.time_remaining = 0
        self.info_time = 0
        self.id = RobotNode._id_counter
        RobotNode._id_counter += 1
        self._same_direction = True
        self.hash_id = hash(start_node) + hash(self.id)

    def retarget(self, new_action, distances):
        if not self.time_remaining == 0:
            raise NotImplementedError('Time remaining must be 0 for now. '
                                      'Cannot switch mid-action')
        if self._cost_to_target <= 0 and self.action is not None:
            # The robot has reached to the node or passed beyond
            self.start = self.action.target_node
            self._start_offset = self._cost_to_target

        # Set the direction of the travel to know robot is traveling inside or outside start node
        self._same_direction = 1 if self.start == new_action.target_node else 0

        self._update_time_to_target(new_action, distances)

        # Store the new action
        self.action = new_action
        self.needs_action = False

    def reset_needs_action(self):
        self.needs_action = True
        self.time_remaining = 0
        self.info_time = 0
        # time_remaining and info_time are set to 0 here because retargeting wouldn't be allowed if time_remaining==0.
        # when a robot finds an object, other robot searching for the same object needs to be re-targeted.

    def advance_time(self, delta_time):
        self.info_time -= delta_time
        self.time_remaining -= delta_time
        self._cost_to_target -= delta_time
        self._start_offset = self._start_offset - self._same_direction * delta_time + \
            + (1 - self._same_direction) * delta_time

        self.info_time = max(0, self.info_time)
        self.time_remaining = max(0, self.time_remaining)

        # What if history says we succeed, so that we end up with negative info time?
        # There's more to consider here, I think. Will need a test to resolve this.

    def _update_time_to_target(self, new_action, distances):
        inter_node_time = distances[(self.start, new_action.target_node)]
        self._cost_to_target = self._same_direction * self._start_offset + \
            (1 - self._same_direction) * abs(self._start_offset) + \
            inter_node_time
        self.time_remaining = max(0, self._cost_to_target + new_action.RS)
        self.info_time = max(0, self._cost_to_target + min(new_action.RS, new_action.RE))

    def copy(self):
        new_robot = RobotNode(self.start)
        new_robot.action = self.action
        new_robot.needs_action = self.needs_action
        new_robot.time_remaining = self.time_remaining
        new_robot.info_time = self.info_time
        new_robot.id = self.id

        new_robot._cost_to_target = self._cost_to_target
        new_robot._start_offset = self._start_offset
        new_robot._same_direction = self._same_direction

        return new_robot

    def __eq__(self, other):
        if not isinstance(other, RobotNode):
            print("Robot object compared with other object")
            return False
        return self.id == other.id

    def __hash__(self):
        return self.hash_id
