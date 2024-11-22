import spot
import itertools
import random
import pytest
import numpy as np
import time

import potlp
from potlp.core import Node


def test_node_raises_error_with_non_list_params():
    _ = Node(())
    _ = Node(("foo",))
    _ = Node(("foo", "bar"))
    _ = Node(("foo", "bar", "foobar"))

    with pytest.raises(ValueError):
        _ = Node("foo")

    with pytest.raises(ValueError):
        _ = Node(("foo"))


def test_spot_parsing():
    """Confirm we can parse a specification without crashing."""
    specification = "F package & F document"
    _ = spot.translate(specification, "BA", "complete")


def gen_random_travel_cost_dict(all_nodes):
    return {
        (n_i, n_o): random.randrange(10, 100) + 1000
        for (n_i, n_o) in itertools.product(all_nodes + ["robot"], repeat=2)
        if not n_i == n_o
    }


def gen_random_subgoal_prop_dict(subgoal_nodes, num_propositions):
    return {
        snode: (1.0 * np.random.random([num_propositions]),
                200 * np.random.random([num_propositions]),
                100 * np.random.random([num_propositions]))
        for snode in subgoal_nodes
    }


@pytest.mark.parametrize(
    "node_props,num_subgoals",
    [((), 4), ((), 8), ((), 1), (("unrelated",), 4), (("unrelated",), 8)],
)
def test_ltl_actions_goal_directed_all_unknown(node_props, num_subgoals):
    specification = "F goal"
    planner = potlp.core.LTLPlanner(specification)

    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    action_dict, _ = planner._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )

    print("All actions:")
    for a, aos in action_dict.items():
        for ao in aos:
            print(ao)

    assert len(action_dict.keys()) == num_subgoals + 1
    assert sum(len(v) for v in action_dict.values()) == num_subgoals**2
    assert all(not a.is_terminal for acts in action_dict.values() for a in acts)
    assert all(
        len(a.unk_dfa_transitions) == 1 for acts in action_dict.values() for a in acts
    )


@pytest.mark.parametrize(
    "node_props,num_subgoals",
    [(("goal",), 4), (("goal", "goal", "goal"), 4), (("goal", "unrelated"), 4)],
)
def test_ltl_actions_goal_directed_goal_seen(node_props, num_subgoals):
    specification = "F goal"
    planner = potlp.core.LTLPlanner(specification)

    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    stime = time.time()
    action_dict, _ = planner._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )

    print(f"All actions (time={time.time()-stime}):")
    for a, aos in action_dict.items():
        for ao in aos:
            print(ao)

    assert len(action_dict.keys()) == num_subgoals + 1
    assert sum(len(list(v)) for v in action_dict.values()) == num_subgoals**2 + (
        num_subgoals + 1
    )
    assert (
        sum(a.is_terminal for acts in action_dict.values() for a in acts)
        == num_subgoals + 1
    )
    assert all(
        len(a.unk_dfa_transitions) == 1
        for acts in action_dict.values()
        for a in acts
        if not a.is_terminal
    )


def test_ltl_actions_and_vs_or_simple_spec():
    """(With no known space nodes) some specifications have more steps.
    An 'AND' spec has more DFA states and so more actions than an 'OR' spec."""
    node_props = ()
    num_subgoals = 2
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    stime = time.time()
    action_dict_and, _ = potlp.core.LTLPlanner("F document & F package")._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    print(f"'AND' actions (time={time.time()-stime}):")
    for a, aos in action_dict_and.items():
        for ao in aos:
            print(ao)

    stime = time.time()
    action_dict_or, _ = potlp.core.LTLPlanner("F document | F package")._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    print(f"'OR' actions (time={time.time()-stime}):")
    for a, aos in action_dict_or.items():
        for ao in aos:
            print(ao)

    assert len(action_dict_and.keys()) > len(action_dict_or.keys())
    assert sum(len(list(v)) for v in action_dict_and.values()) > sum(
        len(list(v)) for v in action_dict_or.values()
    )
    # There is only one way to advance the DFA state for the 'AND' case for each state
    assert any(
        len(a.unk_dfa_transitions) == 1
        for acts in action_dict_and.values()
        for a in acts
        if not a.is_terminal
    )
    # There are multiple ways to advance the DFA state for the 'OR' case for some states
    assert any(
        len(a.unk_dfa_transitions) > 1
        for acts in action_dict_or.values()
        for a in acts
        if not a.is_terminal
    )


def test_potlp_core_path_lengths_sequential_spec():
    p = potlp.core.LTLPlanner("(!foo U bar) & (F foo)")

    node_props = ("foo", "bar")
    num_subgoals = 1
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    stime = time.time()
    action_dict, _ = p._get_actions(known_space_nodes, subgoal_nodes, travel_cost_dict)
    print(f"Actions (time={time.time()-stime}):")
    print(f"Initial State: {p.aut.get_init_state_number()}")
    print(p.semantic_index)
    for a, aos in action_dict.items():
        for ao in aos:
            print(ao)

    # For this spec, no terminal trajectories should be longer than 4 elements,
    # including start, end, and both known-space nodes.
    assert all(
        len(a.node_name_path) <= 4
        for acts in action_dict.values()
        for a in acts
        if a.is_terminal
    )
    # For this spec, no non-terminal trajectories should be longer than 4
    # elements, including start, end, and one of the known-space nodes.
    assert all(
        len(a.node_name_path) <= 3
        for acts in action_dict.values()
        for a in acts
        if not a.is_terminal
    )


@pytest.mark.parametrize("num_subgoals", [2, 4, 8])
def test_potlp_core_remove_non_singluar_transitions(num_subgoals):
    node_props = ()
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    specification = "F document & F package"
    action_dict_all, _ = potlp.core.LTLPlanner(specification)._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    action_dict_singular, _ = potlp.core.LTLPlanner(
        specification, only_singular_transitions=True
    )._get_actions(known_space_nodes, subgoal_nodes, travel_cost_dict)

    print("All actions:")
    for a, aos in action_dict_all.items():
        for ao in aos:
            print(ao)

    print("Singular actions:")
    for a, aos in action_dict_singular.items():
        for ao in aos:
            print(ao)

    # Confirm non-singular transitions are removed
    assert any(
        sum(transition) > 1
        for acts in action_dict_all.values()
        for a in acts
        for transition in a.unk_dfa_transitions
        if not a.is_terminal
    )
    assert all(
        sum(transition) == 1
        for acts in action_dict_singular.values()
        for a in acts
        for transition in a.unk_dfa_transitions
        if not a.is_terminal
    )
    # Confirm that removing the non-singular transitions shrinks the actions.
    assert sum(len(list(v)) for v in action_dict_all.values()) > sum(
        len(list(v)) for v in action_dict_singular.values()
    )


@pytest.mark.parametrize("only_singular_transitions", (False, True))
def test_potlp_core_compute_subgoal_props(only_singular_transitions):
    """Confirm that we can compute the subgoal properties for each action."""
    specification = "(!package U document) & (F package | F box) & (F folder)"
    num_subgoals = 4
    node_props = ("document", "folder")

    # Instantiate the nodes, planner, and travel_cost_dict and get the actions.
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)
    planner = potlp.core.LTLPlanner(
        specification, only_singular_transitions=only_singular_transitions
    )
    stime = time.time()
    action_dict, node_id_dict = planner._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    print(time.time() - stime)

    # In practice, this is computed elsewhere.
    # Stores all estimated properties for each proposition for each subgoal
    subgoal_prop_dict = gen_random_subgoal_prop_dict(
        subgoal_nodes, len(list(planner.semantic_index.keys())))

    # Remap the subgoal prop dictionary (use "node ids" instead of "nodes" as keys)
    subgoal_prop_dict = {
        node_id_dict[snode]: subgoal_prop_vecs
        for snode, subgoal_prop_vecs in subgoal_prop_dict.items()
    }

    # Compute the subgoal properties for each action
    dat = {action: potlp.core.compute_subgoal_props_for_action(action, subgoal_prop_dict)
           for acts in action_dict.values()
           for action in acts
           if not action.is_terminal
           }
    print(f"All actions (count={sum(len(list(v)) for v in action_dict.values())}")
    for a, d in dat.items():
        print(f"{a} | PS={d[0]:.3f}, RS={d[1]:5.1f}, RE={d[2]:5.1f}")


@pytest.mark.parametrize(
    "node_props",
    [(tuple()),
     (("unrelated",)),
     (("folder", "box", "package")),
     (("folder", "box", "package", "unrelated")),
     (("folder", "box", "package", "document")),
     (("folder", "box", "package", "document", "unrelated"))],
)
def test_potlp_core_no_subgoals(node_props):
    """Confirm that we can compute the subgoal properties for each action."""
    specification = "(!package U document) & (F package | F box) & (F folder)"
    node_props = ("folder", "box", "package")

    # Instantiate the nodes, planner, and travel_cost_dict and get the actions.
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = []
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)
    planner = potlp.core.LTLPlanner(specification,)
    stime = time.time()
    action_dict, _ = planner._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    print(time.time() - stime)

    print("All actions:")
    for a, aos in action_dict.items():
        for ao in aos:
            print(ao)



# Added tests for cpp
@pytest.mark.parametrize(
    "node_props",
    [(tuple()),
     (("unrelated", "folder")),
     (("unrelated",)),
     (("folder", "box", "package")),
     (("folder", "box", "package", "unrelated")),
     (("folder", "box", "package", "document")),
     (("folder", "box", "package", "document", "unrelated"))],
)
def test_cpp_and_python_computed_subgoal_props_for_action_match(node_props):
    import potlp_accel
    specification = "(F package | F box) & (F folder)"
    p = potlp.core.LTLPlanner(specification)
    num_subgoals = 2
    # Instantiate the nodes, planner, and travel_cost_dict and get the actions.
    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)
    action_dict, node_id_dict = p._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    # In practice, this is computed elsewhere.
    # Stores all estimated properties for each proposition for each subgoal
    subgoal_prop_dict = gen_random_subgoal_prop_dict(
        subgoal_nodes, len(list(p.semantic_index.keys())))

    # Remap the subgoal prop dictionary (use "node ids" instead of "nodes" as keys)
    subgoal_prop_dict_py = {
        node_id_dict[snode]: subgoal_prop_vecs
        for snode, subgoal_prop_vecs in subgoal_prop_dict.items()
    }

    subgoal_prop_dict_cpp = {
        node_id_dict[snode]: [vec.tolist() for vec in subgoal_prop_vecs]
        for snode, subgoal_prop_vecs in subgoal_prop_dict.items()
    }
    print("Subgoal prop dictionary python:")
    print(subgoal_prop_dict_py)

    print("\nSubgoal prop dictionary cpp:")
    print(subgoal_prop_dict_cpp)
    dat_py = {action: potlp.core.compute_subgoal_props_for_action(action, subgoal_prop_dict_py)
              for acts in action_dict.values()
              for action in acts
              if not action.is_terminal
              }
    dat_cpp = {}
    h_py = {}
    for a, acts in action_dict.items():
        for ao in acts:
            if not ao.is_terminal:
                cpp_action = potlp_accel.Action_cpp(start_state=ao.start_state,
                                                known_state=ao.known_state,
                                                known_space_cost=ao.known_space_cost,
                                                node_name_path=ao.node_name_path,
                                                unk_dfa_state=ao.unk_dfa_state if not ao.is_terminal else 0,
                                                unk_dfa_transitions=[list(a) for a in ao.unk_dfa_transitions] if not ao.is_terminal else [],
                                                is_terminal=ao.is_terminal,
                                                hash_id=1111)

                properties = potlp_accel.compute_subgoal_props_for_action_accel(cpp_action, subgoal_prop_dict_cpp)
                dat_cpp[ao] = properties
    for action, properties in dat_py.items():
        assert properties == dat_cpp[action]
        print(f'python: {properties = }, cpp: {dat_cpp[action] = }')


""" Tests written in cpp """
# (TODO: Abhish) combine two tests below into a single test
def test_function_cpp_update_subgoal_prop_dict():
    import potlp_accel
    result = potlp_accel.test_update_subgoal_prop_dict_accel()
    assert result == True

def test_function_cpp_null_subgoal_prop_dict():
    import potlp_accel
    result = potlp_accel.test_empty_history_subgoal_prop_dict_update_accel()
    assert result == True

def test_function_cpp_updated_ps_with_updated_properties():
    import potlp_accel
    result = potlp_accel.test_updated_ps_with_updated_properties_accel()
    assert result == True

# In this test, state is considered (100, 1)
@pytest.mark.parametrize(
    "history, unk_dfa_transitions, expected_history, success",
    [([], [[0, 0, 1, 0]], [[100, 2, 1]], 1),
     ([], [[1, 1, 0, 0], [0, 0, 1, 0]], [[100, 0, 1], [100, 1, 1], [100, 2, 1]], 1),
     ([[101, 0, 1], [102, 2, 1]], [[1, 1, 0, 0]], [[101, 0, 1], [102, 2, 1], [100, 0, 1], [100, 1, 1]], 1),
     ([], [[0, 0, 1, 0]], [[100, 2, 0]], 0),
     ([], [[1, 1, 0, 0], [0, 0, 1, 0]], [[100, 0, 0], [100, 1, 0], [100, 2, 0]], 0),
     ([[101, 0, 1], [102, 2, 1]], [[1, 1, 0, 0]], [[101, 0, 1], [102, 2, 1], [100, 0, 0], [100, 1, 0]], 0),
    ]
)
def test_function_cpp_add_to_history(history, unk_dfa_transitions, expected_history, success):
    import potlp_accel
    result = potlp_accel.test_add_to_history_accel(history, unk_dfa_transitions, expected_history, success)
    assert result == True


def test_function_cpp_update_action_ps_with_history():
    import potlp_accel
    subgoal_prop_dict_cpp = {100: [[0.3, 0.2], [20, 20], [20, 20]],
                             101: [[0.8, 0.7], [30, 30], [30, 30]]}
    history = [[100, 0, 1], [101, 1, 0]]

    action_1_cpp = potlp_accel.Action_cpp(start_state=(101, 1),
                                    known_state=(100, 1),
                                    known_space_cost=10,
                                    node_name_path=[(101, 1), (100, 1)],
                                    unk_dfa_state=0,
                                    unk_dfa_transitions=[[1, 0],],
                                    is_terminal=False,
                                    hash_id=1)
    action_1_py = potlp.core.Action(start_state=(101, 1),
                                known_state=(100, 1),
                                known_space_cost=10,
                                node_name_path=[(101, 1), (100, 1)],
                                unk_dfa_state=0,
                                unk_dfa_transitions={(1, 0),},
                                is_terminal=False)
    PS_1, RS_1, RE_1 = potlp_accel.test_get_ps_rs_re_with_history_accel(subgoal_prop_dict_cpp,
                                                                history,
                                                                action_1_cpp)

    updated_subgoal_prop_dict_py_1 = {100: (np.array([1, 0.2]), np.array([20, 20]), np.array([20, 20])),
                                101: (np.array([0.8, 0]), np.array([30, 30]), np.array([30, 30]))}

    PS, RS, RE = potlp.core.compute_subgoal_props_for_action(action_1_py, updated_subgoal_prop_dict_py_1)

    assert PS_1 == PS
    assert RS_1 == RS
    assert RE_1 == RE

    action_2_cpp = potlp_accel.Action_cpp(start_state=(100, 1),
                                    known_state=(101, 1),
                                    known_space_cost=10,
                                    node_name_path=[(100, 1), (101, 1)],
                                    unk_dfa_state=0,
                                    unk_dfa_transitions=[[0, 1],],
                                    is_terminal=False,
                                    hash_id=2)
    action_2_py = potlp.core.Action(start_state=(100, 1),
                                known_state=(101, 1),
                                known_space_cost=10,
                                node_name_path=[(100, 1), (101, 1)],
                                unk_dfa_state=0,
                                unk_dfa_transitions={(0, 1),},
                                is_terminal=False)

    PS_2, RS_2, RE_2= potlp_accel.test_get_ps_rs_re_with_history_accel(subgoal_prop_dict_cpp,
                                                                    history,
                                                                    action_2_cpp)

    updated_subgoal_prop_dict_py_2 = {100: (np.array([1, 0.2]), np.array([20, 20]), np.array([20, 20])),
                                101: (np.array([0.8, 0]), np.array([30, 30]), np.array([30, 30]))}

    PS, RS, RE = potlp.core.compute_subgoal_props_for_action(action_2_py, updated_subgoal_prop_dict_py_2)

    assert PS_2 == PS
    assert RS_2 == RS
    assert RE_2 == RE


# Test for POTLP tree
@pytest.mark.parametrize("specification",
                         ["F goal",
                          "(!goal U package) & F goal",
                          "(!package U document) & (F package | F box) & (F folder)",])
@pytest.mark.parametrize(
    " node_props,num_subgoals",
    [((), 2),
    (('goal', ), 2),
    (('goal','package'), 2),
    (('goal','package','document'), 7),
    ],
)
def test_potlp_pipeline_works_without_error(specification, node_props, num_subgoals):
    planner = potlp.core.LTLPlanner(specification)

    known_space_nodes = [Node(props=(ps,), is_subgoal=False) for ps in node_props]
    subgoal_nodes = [Node(is_subgoal=True) for _ in range(num_subgoals)]
    travel_cost_dict = gen_random_travel_cost_dict(known_space_nodes + subgoal_nodes)

    action_dict, node_id_dict = planner._get_actions(
        known_space_nodes, subgoal_nodes, travel_cost_dict
    )
    # In practice, this is computed elsewhere.
    # Stores all estimated properties for each proposition for each subgoal
    subgoal_prop_dict = gen_random_subgoal_prop_dict(
        subgoal_nodes, len(list(planner.semantic_index.keys())))

    # The initial state for robot to start
    robot_node_id = 1 + len(subgoal_nodes) + len(known_space_nodes) + 200
    initial_dfa_state = planner.aut.get_init_state_number()
    initial_state = (robot_node_id, initial_dfa_state)

    # Find the best action
    best_action = potlp.find_best_action_accel(initial_state=initial_state,
                                                action_dict=action_dict,
                                                subgoal_prop_dict=subgoal_prop_dict,
                                                node_id_dict=node_id_dict)
    print(best_action)

@pytest.mark.parametrize("ps_vec, unk_dfa_transitions", [
        (np.array([0, 0, 0]), [[1, 0, 0], [0, 1, 1], [0, 1, 0]]),
        (np.array([0, 0.2, 0]), [[1, 0, 0], [0, 1, 1], [0, 1, 0]]),
        (np.array([0.2, 0.3, 0.8]), [[1, 0, 0], [0, 1, 1], [0, 1, 0]]),
        (np.array([1, 0.2]), [[1, 0],]),
        (np.array([0.8, 0]), [[1, 0],]),
        (np.array([1, 0.2]), [[0, 1],]),
        (np.array([0.8, 0]), [[0, 1],]),
        ])
def test_PS_per_transition(ps_vec, unk_dfa_transitions):
    import potlp_accel
    result_cpp = potlp_accel.get_PS_per_transition_accel(ps_vec, unk_dfa_transitions)
    result_py = np.array([np.prod((ps_vec * t)[np.array(t) == 1])
                                  for t in unk_dfa_transitions])
    print(result_cpp, result_py)
    assert tuple(result_cpp) == tuple(result_py)
