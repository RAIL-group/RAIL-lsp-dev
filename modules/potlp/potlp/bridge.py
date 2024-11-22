import random
import potlp_accel

def find_best_action_accel(initial_state, action_dict, subgoal_prop_dict, node_id_dict, num_iterations=1000):
    subgoal_prop_dict_cpp = {
        # node_id_dict[snode]: [vec for vec in subgoal_prop_vecs]
        # node_id_dict[snode]: [vec for vec in subgoal_prop_vecs] if snode not in node_id_dict else node_id_dict[snode]
        # for snode, subgoal_prop_vecs in subgoal_prop_dict.items()
    }
    for snode, subgoal_prop_vecs in subgoal_prop_dict.items():
        if snode in node_id_dict:
            key = node_id_dict[snode]
            if key not in subgoal_prop_dict_cpp:
                subgoal_prop_dict_cpp[key] = [vec for vec in subgoal_prop_vecs]
            else:
                subgoal_prop_dict_cpp[key].extend([vec for vec in subgoal_prop_vecs])

    action_dict_cpp = {}
    h_cpp = {}
    all_ao = []
    for a, aos in action_dict.items():
        action_cpp = []
        for ao in aos:
            hash = random.getrandbits(16)
            h_cpp[hash] = ao
            # Test the cpp action class
            cpp_action = potlp_accel.Action_cpp(start_state=ao.start_state,
                                                known_state=ao.known_state,
                                                known_space_cost=ao.known_space_cost,
                                                node_name_path=ao.node_name_path,
                                                unk_dfa_state=ao.unk_dfa_state if not ao.is_terminal else 0,
                                                unk_dfa_transitions=[list(a) for a in ao.unk_dfa_transitions] if not ao.is_terminal else [],
                                                is_terminal=ao.is_terminal,
                                                hash_id=hash)
            string = f'Action:{a}:Start_state:{cpp_action.start_state}, Known_state:{cpp_action.known_state}, Known_space_cost:{cpp_action.known_space_cost}, Node_name_path:{cpp_action.node_name_path}, Unk_dfa_state:{cpp_action.unk_dfa_state}, Unk_dfa_transitions:{cpp_action.unk_dfa_transitions}, Is_terminal:{cpp_action.is_terminal}, Hash_id:{cpp_action.hash_id}'
            all_ao.append(string)
            action_cpp.append(cpp_action)
        action_dict_cpp[a] = action_cpp
    print(initial_state)
    print("---------------------------------------------------------")
    with open("/data/all_ao.txt", "w") as f:
        f.write("\n".join(all_ao))
        f.write("\nInitial_state: " + str(initial_state))
        f.write("\nSubgoal_prop_dict: " + str(subgoal_prop_dict_cpp))

    best_action_cpp = potlp_accel.find_best_action(initial_state, action_dict_cpp, subgoal_prop_dict_cpp, num_iterations)
    best_action_py = h_cpp[best_action_cpp.hash_id]
    return best_action_py