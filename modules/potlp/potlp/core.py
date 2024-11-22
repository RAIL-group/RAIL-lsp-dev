import common
import copy
import itertools
import gridmap
import networkx as nx
import numpy as np
import spot
import lsp
from collections import namedtuple
from gridmap.constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                               OBSTACLE_THRESHOLD)

def _node_satisfies_transition_array(n_feat, transitions):
    """Helper: Returns whether a particular node (feat array) allows
    a specified transition (edge, as feat array)."""
    n_feat = np.array(n_feat)
    return any(
        (n_feat[transition == 1] == 1).all()
        and (n_feat[transition == -1] == 0).all()
        for transition in transitions
    )


def get_path_from_node_path(planning_grid, robot_pose, node_path):
    path_elements = []

    for node_start, node_end in zip(node_path[:-1], node_path[1:]):
        if isinstance(node_start, str) and node_start == "robot":
            ns_pos = (int(robot_pose.x), int(robot_pose.y))
        else:
            ns_pos = node_start.position

        if isinstance(node_end, str) and node_end == "end":
            continue
        else:
            ne_pos = node_end.position

        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, ns_pos)
        did_plan, path = get_path(ne_pos)
        path_elements.append(path)

    return np.concatenate(path_elements, axis=1)


class Node(object):
    def __init__(self, props=None, is_subgoal=False, position=None, subgoal=None):
        if props is not None and isinstance(props, str):
            raise ValueError(
                "'props' must be a list or tuple of strings. "
                f"Instead of 'props={props}' or 'props=({props})' try: "
                f"'props=({props},)' (note the comma, which makes a tuple)."
            )
        self.props = props
        self.position = position
        self.subgoal = subgoal

    def str(self):
        # return self.props + "  Position" + "("+self.position.x+ ", " + self.position.y+")"
        return self.props + self.position
    def subgoal(self):
        return self.subgoal

class Action(object):
    def __init__(
        self,
        start_state,
        known_state,
        known_space_cost,
        node_name_path,
        unk_dfa_state=None,
        unk_dfa_transitions=None,
        is_terminal=False,
    ):
        if not is_terminal and (unk_dfa_state is None or unk_dfa_transitions is None):
            raise ValueError("If not terminal, state and transitions required.")

        self.start_state = start_state
        self.known_state = known_state
        self.known_space_cost = known_space_cost
        self.node_name_path = node_name_path
        self.unk_dfa_state = unk_dfa_state
        self.unk_dfa_transitions = unk_dfa_transitions
        self.is_terminal = is_terminal

    def __str__(self):
        if self.is_terminal:
            return (
                f"Action [terminal]: {self.start_state}->{self.known_state} "
                + f"(cost={self.known_space_cost}, path={self.node_name_path})"
            )
        else:
            return (
                f"Action: {self.start_state}->{self.known_state}->{self.unk_dfa_state} "
                + f"via {self.unk_dfa_transitions} "
                + f"(cost={self.known_space_cost}, path={self.node_name_path})"
            )

    def to_node_path(self, node_id_dict):
        id_node_dict = {v: k for k, v in node_id_dict.items()}
        return [id_node_dict[node_id] for node_id, _ in self.node_name_path]


class LTLPlanner(object):
    def __init__(
        self, specification, complete=False, only_singular_transitions=False
    ):
        self.only_singular_transitions = only_singular_transitions
        if complete:
            self.aut = spot.translate(specification, "BA", "complete")
        else:
            self.aut = spot.translate(specification, "BA")

        self.bdict = self.aut.get_dict()

        self.accepting_states = set(
            t.dst
            for s in range(self.aut.num_states())
            for t in self.aut.out(s)
            if t.acc.count() > 0 and t.dst == t.src
        )
        self.semantic_index = {
            prop: num for num, prop in enumerate([ap.ap_name() for ap in self.aut.ap()])
        }

        # Loop through all possible in/out DFA states and get transitions
        # Transitions stored so that +1 if must be true and 0 else.
        self.action_edge_dict = {
            (sin, sout): set(tuple((t == 1).astype(int)) for t in transitions)
            for sin in range(self.aut.num_states())
            for transitions, sout in self.get_transitions_from_state(sin)
        }
        if self.only_singular_transitions:
            # Keep only transitions with one true prop
            self.action_edge_dict = {
                key: [t for t in transitions if sum(t) == 1]
                for key, transitions in self.action_edge_dict.items()
            }

    def _node_to_prop_array(self, node):
        return [int(prop in node.props) for prop in self.semantic_index.keys()]

    def get_transitions_from_state(self, s):
        """Finds the valid transitions leaving a states s as defined by the edge label."""
        return [
            (self.props_to_array(spot.bdd_format_formula(self.bdict, t.cond)), t.dst)
            for t in self.aut.out(s)
        ]

    def get_init_dfa_state(self):
        return self.aut.get_init_state_number()

    def update_dfa_state(self, dfa_state, node):
        node_feat = self._node_to_prop_array(node)

        for transition_array, dfa_state_end in self.get_transitions_from_state(dfa_state):
            if _node_satisfies_transition_array(node_feat, transition_array):
                return dfa_state_end

        # No transition possible
        return dfa_state

    def is_dfa_state_accepting(self, dfa_state):
        return dfa_state in self.accepting_states

    def props_to_array(self, props_in):
        """Transforms the string of proposition values to a numpy array"""
        # For each formula in props_in, splits that proposition into chunks of based on
        # ors. For each segment of the ors,
        # finds all truth values that satisfy the ands. Assumes the formulas are like (a & b) | (c & d)
        # TODO(@apacheck): Change this to use the bdd from spot
        n_props = len(self.semantic_index)

        # QUESTION: I'm unsure why this succeeds.
        if props_in[0] == "1":
            return np.zeros((1, n_props), dtype=int)
        expand_props = list(self.semantic_index.keys())

        # QUESTION: is this needed?
        if isinstance(props_in, str):
            props_in = [props_in]

        props_return = np.array([[]])
        for props_t in props_in:
            props_or_split = props_t.split(" | ")
            for props_or in props_or_split:
                # To remove the parentheses
                props_or = props_or.replace("(", "").replace(")", "")
                props_split = props_or.split(" & ")
                props_truth = np.zeros((1, n_props), dtype=int)
                props_all = copy.deepcopy(expand_props)
                for p in props_split:
                    if p[0] == "!":
                        props_truth[:, expand_props.index(p[1:])] = -1
                        props_all.remove(p[1:])
                    else:
                        props_truth[:, expand_props.index(p)] = 1
                        props_all.remove(p)

                # Creates expansion of combinations of booleans using itertools
                # package and then replaces all the zero
                # rows with the created combinations
                if props_return.size > 0:
                    props_return = np.concatenate([props_return, props_truth], axis=0)
                else:
                    props_return = np.copy(props_truth)

        return np.unique(props_return, axis=0)

    def _get_actions(
        self,
        known_space_nodes,
        subgoal_nodes,
        travel_cost_dict,
        dfa_start=None,
    ):

        if dfa_start is None:
            dfa_start = self.aut.get_init_state_number()

        # Rename the node arrays to use integers
        end_node_id = len(subgoal_nodes) + len(known_space_nodes) + 200
        robot_node_id = end_node_id + 1
        ks_prop_arrays = [
            self._node_to_prop_array(ks_node) for ks_node in known_space_nodes
        ]
        ks_node_prop_dict = {
            num: ks_prop_array
            for num, ks_prop_array in enumerate(ks_prop_arrays)
            if sum(ks_prop_array) > 0
        }
        # Remove known space nodes that don't have relevant props
        known_space_nodes = [ks_node for ks_node, ks_prop_array
                             in zip(known_space_nodes, ks_prop_arrays)
                             if sum(ks_prop_array) > 0]
        subgoal_names = [
            ii + len(known_space_nodes) + 100 for ii in range(len(subgoal_nodes))
        ]
        node_id_pairs = (
            list(zip(subgoal_nodes, subgoal_names))
            + list(zip(known_space_nodes, ks_node_prop_dict.keys()))
            + [("robot", robot_node_id)]
        )
        travel_cost_dict = {
            (name_a, name_b): travel_cost_dict[(node_a, node_b)]
            for ((node_a, name_a), (node_b, name_b)) in itertools.product(
                node_id_pairs, repeat=2
            )
            if (node_a, node_b) in travel_cost_dict.keys()
        }

        # We loop through all states reachable from the current state
        visited_states = set()
        reachable_states = set([dfa_start])
        known_space_state_transitions = nx.DiGraph()
        all_dfa_transitions = dict()

        while reachable_states - visited_states:
            s_new = (reachable_states - visited_states).pop()
            visited_states.add(s_new)
            transitions = self.get_transitions_from_state(s_new)
            all_dfa_transitions[s_new] = set([s_new])

            for transition_array, s_end in transitions:
                reachable_states.add(s_end)

                # Do not add self-edges
                if s_new == s_end:
                    continue

                transitioning_nodes = [
                    node_id
                    for node_id, node_feat in ks_node_prop_dict.items()
                    if _node_satisfies_transition_array(node_feat, transition_array)
                ]

                # If no nodes transition to a new state, continue
                if not transitioning_nodes:
                    continue

                all_dfa_transitions[s_new].add(s_end)

                # Add possible traversals between the known-space state
                # transitions that allow DFA transition s_new->s_end
                known_space_state_transitions.add_weighted_edges_from(
                    [
                        (
                            (node_start, s_new),
                            (node_end, s_end),
                            travel_cost_dict[(node_start, node_end)],
                        )
                        for node_start, node_end in itertools.product(
                            ks_node_prop_dict.keys(), transitioning_nodes
                        )
                        if not node_start == node_end
                    ]
                )

                # Adds edges from subgoals (and robot) to each of the nodes
                # that allow DFA transition s_new->s_end
                known_space_state_transitions.add_weighted_edges_from(
                    [
                        (
                            (sg_start, s_new),
                            (node_end, s_end),
                            travel_cost_dict[(sg_start, node_end)],
                        )
                        for sg_start, node_end in itertools.product(
                            subgoal_names + [robot_node_id], transitioning_nodes
                        )
                    ]
                )

        # Finally, add transitions the subgoal nodes and their edges
        known_space_state_transitions.add_weighted_edges_from(
            ((knode, dfa_state), (snode, dfa_state), travel_cost_dict[(snode, knode)])
            for snode, knode in itertools.product(
                subgoal_names, ks_node_prop_dict.keys()
            )
            for dfa_state in visited_states
        )
        known_space_state_transitions.add_weighted_edges_from(
            (
                (ssnode, dfa_state),
                (senode, dfa_state),
                travel_cost_dict[(ssnode, senode)],
            )
            for ssnode, senode in itertools.product(subgoal_names, repeat=2)
            for dfa_state in visited_states
            if not ssnode == senode
        )
        known_space_state_transitions.add_weighted_edges_from(
            (
                (robot_node_id, dfa_state),
                (senode, dfa_state),
                travel_cost_dict[(robot_node_id, senode)],
            )
            for senode in subgoal_names
            for dfa_state in visited_states
        )
        known_space_state_transitions.add_weighted_edges_from(
            ((knode, dfa_state), (end_node_id, dfa_state), 0)
            for knode in list(ks_node_prop_dict.keys()) + [robot_node_id]
            for dfa_state in visited_states
            if dfa_state in self.accepting_states
        )

        all_actions = dict()
        for start_sub, dfa_start in itertools.product(
            subgoal_names + [robot_node_id], visited_states
        ):
            if dfa_start in self.accepting_states:
                continue

            if not known_space_state_transitions.has_node((start_sub, dfa_start)):
                all_actions[(start_sub, dfa_start)] = []
                continue

            # Computes all shortest paths from a given start state
            path_dict = nx.shortest_path(
                known_space_state_transitions, (start_sub, dfa_start), weight="weight"
            )


            # Get all the action "pieces" involving known space
            action_ins = [
                (
                    target,
                    path,
                    nx.path_weight(
                        known_space_state_transitions, path, weight="weight"
                    ),
                )
                for target, path in path_dict.items()
                if target[0] in subgoal_names and len(path) > 1
            ]
            action_term = [
                (
                    target,
                    path,
                    nx.path_weight(
                        known_space_state_transitions, path, weight="weight"
                    ),
                )
                for target, path in path_dict.items()
                if target[0] == end_node_id and len(path) > 1
            ]

            # Update dictionary that stores the set of actions for start state
            # Loop through 'action_ins' and transitions they could allow
            # in unknown space once reached.
            all_actions[(start_sub, dfa_start)] = [
                Action(
                    start_state=(start_sub, dfa_start),
                    known_state=target_state_known,
                    known_space_cost=cost,
                    node_name_path=path,
                    unk_dfa_state=dfa_new,
                    unk_dfa_transitions=self.action_edge_dict[
                        (target_state_known[1], dfa_new)
                    ],
                )
                for target_state_known, path, cost in action_ins
                for dfa_new in [t.dst for t in self.aut.out(target_state_known[1])]
                if not dfa_new == target_state_known[1]
                and self.action_edge_dict[(target_state_known[1], dfa_new)]
            ] + [
                Action(
                    start_state=(start_sub, dfa_start),
                    known_state=target_state_known,
                    known_space_cost=cost,
                    node_name_path=path,
                    is_terminal=True,
                )
                for target_state_known, path, cost in action_term
            ]

        node_id_dict = {
            node: node_id
            for (node, node_id) in node_id_pairs + [("end", end_node_id)]
        }

        return all_actions, node_id_dict

    def __str__(self):
        bdict = self.aut.get_dict()
        out = ""
        out += "Acceptance: {}\n".format(self.aut.get_acceptance())
        out += "Number of sets: {}\n".format(self.aut.num_sets())
        out += "Number of states: {}\n".format(self.aut.num_states())
        out += "Initial states: {}\n".format(self.aut.get_init_state_number())
        out += "Atomic propositions:"
        for ap in self.aut.ap():
            out += " {} (={})".format(ap, bdict.varnum(ap))
        out += "\n"
        # Templated methods are not available in Python, so we cannot
        # retrieve/attach arbitrary objects from/to the automaton.  However the
        # Python bindings have get_name() and set_name() to access the
        # "automaton-name" property.
        name = self.aut.get_name()
        if name:
            out += "Name: {}\n".format(name)
        out += "Deterministic: {}\n".format(
            self.aut.prop_universal() and self.aut.is_existential()
        )
        out += "Unambiguous: {}\n".format(self.aut.prop_unambiguous())
        out += "State-Based Acc: {}\n".format(self.aut.prop_state_acc())
        out += "Terminal: {}\n".format(self.aut.prop_terminal())
        out += "Weak: {}\n".format(self.aut.prop_weak())
        out += "Inherently Weak: {}\n".format(self.aut.prop_inherently_weak())
        out += "Stutter Invariant: {}\n".format(self.aut.prop_stutter_invariant())

        for s in range(0, self.aut.num_states()):
            out += "State {}:\n".format(s)
            for t in self.aut.out(s):
                out += "  edge({} -> {})\n".format(t.src, t.dst)
                out += "    label = {}\n".format(spot.bdd_format_formula(bdict, t.cond))
                out += "    acc sets = {}\n".format(t.acc)

        return out


def compute_subgoal_props_for_action(action, subgoal_prop_dict):
    # Get the subgoal property vectors for the subgoal of interest
    ps_vec, rs_vec, re_vec = subgoal_prop_dict[action.known_state[0]]

    PS_per_transition = np.array([np.prod((ps_vec * t)[np.array(t) == 1])
                                  for t in action.unk_dfa_transitions])
    RS_per_transition = np.array([max(rs_vec[np.array(t) == 1])
                                  for t in action.unk_dfa_transitions])
    RE_per_transition = np.array([max(re_vec[np.array(t) == 1])
                                  for t in action.unk_dfa_transitions])

    PS = 1 - np.prod(1 - PS_per_transition)
    if (np.sum(PS_per_transition) != 0):
        RS = np.average(RS_per_transition, weights=PS_per_transition)
    else:
        RS = 0
    if (np.sum(1 - PS_per_transition) != 0):
        RE = np.average(RE_per_transition, weights=1 - PS_per_transition)
    else:
        RE = 0

    return PS, RS, RE


def get_travel_cost_dict(grid, all_nodes, robot_pose):
    NodeWithPoint = namedtuple('NodeWithPoint', ['node', 'point'])
    node_with_point_list = [
        NodeWithPoint(node=node,
                      point=node.position)
        for node in all_nodes
    ] + [NodeWithPoint(node='robot', point=[int(robot_pose.x), int(robot_pose.y)])]

    all_points = np.stack([nwp.point for nwp in node_with_point_list], axis=1)
    occupancy_grid = np.copy(grid)
    occupancy_grid[occupancy_grid == UNOBSERVED_VAL] = COLLISION_VAL
    occupancy_grid[all_points[0, :],
                   all_points[1, :]] = FREE_VAL

    # I only need the upper triangular block of the pairs. This means that I
    # don't need the final fwp_1 (since it would only be comparing against
    # itself) and I use the enumerate function to select only a subset of the
    # fwp_2 entries.
    travel_cost_dict = dict()
    for ind, nwp_1 in enumerate(node_with_point_list[:-1]):
        # Compute the cost grid for the first frontier
        start = nwp_1.point
        cost_grid = gridmap.planning.compute_cost_grid_from_position(
            occupancy_grid,
            start=start,
            use_soft_cost=False,
            only_return_cost_grid=True)
        for nwp_2 in node_with_point_list[ind + 1:]:
            nn_set = frozenset([nwp_1.node, nwp_2.node])
            cost = cost_grid[nwp_2.point[0], nwp_2.point[1]].min()
            # travel_cost_dict[nn_set] = cost
            travel_cost_dict[(nwp_1.node, nwp_2.node)] = cost
            travel_cost_dict[(nwp_2.node, nwp_1.node)] = cost

    return travel_cost_dict

def get_known_props_for_node(known_space_node_poses, all_frontiers, subgoal_node, known_grid, observed_grid, robot):
    node_poses = known_space_node_poses.copy()
    masked_observed_grid = lsp.core.mask_grid_with_frontiers(
        observed_grid, all_frontiers, do_not_mask=subgoal_node.subgoal)
    masked_mixed_grid = masked_observed_grid.copy()
    masked_mixed_grid[known_grid == COLLISION_VAL] = COLLISION_VAL
    # Use masked mixed grid here
    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        masked_mixed_grid, [robot.pose.x, robot.pose.y], use_soft_cost=True)
    ''' If there are more than 1 pose, minimum success cost must be returned'''
    all_RS = []
    all_PS = []
    all_RE = []
    for pose in node_poses:
        '''If the pose is in known space, do not consider it'''
        if observed_grid[pose[0], pose[1]] == FREE_VAL:
            continue

        did_plan, path = get_path([pose[0], pose[1]], do_sparsify=True, do_flip=True)
        if did_plan:
            all_PS.append(1.0)
            all_RS.append(common.compute_path_length(path)) # not the complete path but from unseen to goal
            all_RE.append(0)
        else:
            all_PS.append(0.0)
            all_RS.append(0)
            # find exploration cost
            min_cost = np.min(cost_grid[masked_mixed_grid == UNOBSERVED_VAL])
            cost_grid[cost_grid > 1e8] = 0
            max_cost = np.max(cost_grid[masked_mixed_grid == UNOBSERVED_VAL])
            if min_cost > 1e8 or max_cost > 1e8:
                exploration_cost = 0
            else:
                exploration_cost = (max_cost - min_cost) * 2
            all_RE.append(exploration_cost)

    # return the ps, rs, re for the index where there is minimum rs
    if len(all_PS) != 0:
        min_rs_index = np.argmin(all_RS)
        return all_PS[min_rs_index], all_RS[min_rs_index], all_RE[min_rs_index]
    else:
        min_cost = np.min(cost_grid[masked_mixed_grid == UNOBSERVED_VAL])
        cost_grid[cost_grid > 1e8] = 0
        max_cost = np.max(cost_grid[masked_mixed_grid == UNOBSERVED_VAL])
        if min_cost > 1e8 or max_cost > 1e8:
            exploration_cost = 0
        else:
            exploration_cost = (max_cost - min_cost) * 2

        return 0, 0, exploration_cost


def get_known_subgoal_props_dict_updated(inflated_known_grid, inflated_observed_grid, known_space_nodes, subgoal_nodes, robot):
    subgoal_prop_dict = {}
    known_space_nodes_poses= {}
    # Find the list of poses for known space nodes
    for known_space_node in known_space_nodes:
        if known_space_node.props[0] not in known_space_nodes_poses:
            known_space_nodes_poses[known_space_node.props[0]] = [(known_space_node.position[0],
                                                                     known_space_node.position[1])]
        else:
            known_space_nodes_poses[known_space_node.props[0]].append((known_space_node.position[0],
                                                                        known_space_node.position[1]))
    all_frontiers = {subgoal_node.subgoal for subgoal_node in subgoal_nodes}
    for subgoal_node in subgoal_nodes:
        PS_vec = []
        RS_vec = []
        RE_vec = []
        already_calculated_objects = []
        for known_space_node in known_space_nodes:
            node_object = known_space_node.props[0]
            if node_object not in already_calculated_objects:
                already_calculated_objects.append(node_object)
                PS, RS, RE = get_known_props_for_node(
                                        known_space_nodes_poses[node_object],
                                        all_frontiers,
                                        subgoal_node,
                                        inflated_known_grid,
                                        inflated_observed_grid,
                                        robot)
            else:
                continue
            PS_vec.append(PS)
            RS_vec.append(RS)
            RE_vec.append(RE)

        subgoal_prop_dict[subgoal_node] = np.array([PS_vec, RS_vec, RE_vec])
    return subgoal_prop_dict
