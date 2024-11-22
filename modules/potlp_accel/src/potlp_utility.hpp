#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <memory>
#include <tuple>

#ifndef _TYPEDEF
#define _TYPEDEF
#include "typedef.h"
#endif

struct Action {
    std::tuple<int, int> start_state;
    std::tuple<int, int> known_state;
    double known_space_cost;
    std::vector<std::tuple<int, int>> node_name_path;
    int unk_dfa_state;
    structure::UnkDfaTransitions unk_dfa_transitions;
    bool is_terminal;
    int64_t hash_id;

    Action(std::tuple<int, int> start_state,
           std::tuple<int, int> known_state,
           double known_space_cost,
           std::vector<std::tuple<int, int>> node_name_path,
           int unk_dfa_state,
           structure::UnkDfaTransitions unk_dfa_transitions,
           bool is_terminal,
           int64_t hash_id) {
        this->start_state = start_state;
        this->known_state = known_state;
        this->known_space_cost = known_space_cost;
        this->node_name_path = node_name_path;
        this->is_terminal = is_terminal;
        this->unk_dfa_state = unk_dfa_state;
        this->unk_dfa_transitions = unk_dfa_transitions;
        this->hash_id = hash_id;
    }

    uint32_t get_hash() const {return hash_id;}
};
typedef std::shared_ptr<Action> ActionPtr;
typedef std::map<std::tuple<int, int>, std::vector<ActionPtr>> ActionDict;

class TreeActionAndSubgoalProps {
 public:
    ActionDict action_dict;
    structure::SubgoalPropDict subgoal_prop_dict;
    TreeActionAndSubgoalProps(ActionDict action_dict,
                            structure::SubgoalPropDict subgoal_prop_dict) {
        this->action_dict = action_dict;
        this->subgoal_prop_dict = subgoal_prop_dict;
    }
    // write a function to print action_dict
    void print_action_dict() {
        for (auto const& x : action_dict) {
            std::cout << "State: " << std::get<0>(x.first) << " " << std::get<1>(x.first) << std::endl;
            for (auto const& y : x.second) {
                std::cout << "Action: " << std::get<0>(y->start_state) << " " <<
                 std::get<1>(y->start_state) << " " << std::get<0>(y->known_state) << " " <<
                 std::get<1>(y->known_state) << " " << y->known_space_cost << " " << y->is_terminal << std::endl;
            }
        }
    }

    // write a function to print subgoal_prop_dict
    void print_subgoal_prop_dict(){
        for (auto const& x : subgoal_prop_dict) {
            std::cout << "State: " << x.first << std::endl;
            for (auto const& y : x.second) {
                for (auto const& z : y) {
                    std::cout << z << " ";
                }
                std::cout << std::endl;
            }
        }
    }
};

int find_action_index(const ActionPtr &action,
                      const std::vector<ActionPtr> &actions) {
    int action_index = -1;
    for (int i = 0; i < actions.size(); i++) {
        if (actions[i]->get_hash() == action->get_hash()) {
            action_index = i;
            break;
        }
    }
    return action_index;
}


std::vector<double> get_PS_per_transition(
        const std::vector<double>& ps_vec,
        const structure::UnkDfaTransitions& unk_dfa_transitions) {
    std::vector<double> PS_per_transition;
    for (const auto& unk_dfa_transition : unk_dfa_transitions) {
        std::vector<double> result;
        bool is_non_zero = true;
        double acc_product = 1.0;
        for (int i = 0; i < unk_dfa_transition.size(); i++) {
            if (unk_dfa_transition[i] == 1) {
                double product = ps_vec[i] * unk_dfa_transition[i];
                if (product > 0.0001) {
                    is_non_zero = false;
                }
                acc_product *= product;
            }
        }
        if (is_non_zero) {
            PS_per_transition.push_back(0.0);
            continue;
        }
        PS_per_transition.push_back(acc_product);
    }
    return PS_per_transition;
}

std::vector<double> get_Reward_per_transition(const std::vector<double>& reward_vec,
                                              const structure::UnkDfaTransitions& unk_dfa_transitions) {
    std::vector<double> Reward_per_transition;
    for (const auto& unk_dfa_transition : unk_dfa_transitions) {
        std::vector<double> reward_if_transition;
        for (int i = 0; i < reward_vec.size(); i++) {
            if (unk_dfa_transition[i] == 1) {
                reward_if_transition.push_back(reward_vec[i]);
            }
        }
        // find max reward
        double max_reward = 0.0;
        for (const auto& r : reward_if_transition) {
            if (r > max_reward) {
                max_reward = r;
            }
        }
        Reward_per_transition.push_back(max_reward);
    }
    return Reward_per_transition;
}


std::tuple<double, double, double> compute_subgoal_props_for_action(const ActionPtr &action,
                                structure::SubgoalPropDict subgoal_prop_dict) {
    // If the action is terminal action
    if (action->is_terminal) {
        return std::make_tuple(1.0, 0.0, 0.0);
    }
    std::vector<double> ps_vec = subgoal_prop_dict[std::get<0>(action->known_state)][0];
    std::vector<double> rs_vec = subgoal_prop_dict[std::get<0>(action->known_state)][1];
    std::vector<double> re_vec = subgoal_prop_dict[std::get<0>(action->known_state)][2];

    std::vector<double> PS_per_transition = get_PS_per_transition(ps_vec, action->unk_dfa_transitions);
    std::vector<double> RS_per_transition = get_Reward_per_transition(rs_vec, action->unk_dfa_transitions);
    std::vector<double> RE_per_transition = get_Reward_per_transition(re_vec, action->unk_dfa_transitions);

    int n = PS_per_transition.size();
    double PS = 1.0;
    for (const auto & num : PS_per_transition) {
        PS *= (1 - num);
    }
    PS = 1 - PS;

    double RS = 0;
    double n1 = 0;
    for (int i = 0; i < n; i++) {
        RS += PS_per_transition[i] * RS_per_transition[i];
        n1 += PS_per_transition[i];
    }
    if (n1 != 0) {
        RS = RS / n1;
    } else {
        RS = 0;
    }

    double RE = 0;
    double n2 = 0;
    for (int i = 0; i < n; i++) {
        RE += (1 - PS_per_transition[i]) * RE_per_transition[i];
        n2 += (1 - PS_per_transition[i]);
    }
    if (n2 != 0) {
        RE = RE / n2;
    } else {
        RE = 0;
    }    // std::cout << "PS: " << PS << " RS: " << RS << " RE: " << RE << std::endl;
    return std::make_tuple(PS, RS, RE);
}

structure::SubgoalPropDict get_updated_subgoal_prop_dictionary(
        structure::SubgoalPropDict subgoal_prop_dict,
        structure::History history) {
    structure::SubgoalPropDict updated_subgoal_prop_dictionary = subgoal_prop_dict;
    for (auto hist : history) {
        int subgoal = std::get<0>(hist);
        int object_index = std::get<1>(hist);
        int object_available = std::get<2>(hist);
        updated_subgoal_prop_dictionary[subgoal][0][object_index] = object_available;
    }
    return updated_subgoal_prop_dictionary;
}

std::tuple<double, double, double>  compute_updated_subgoal_props_with_history(
        const ActionPtr &action,
        structure::SubgoalPropDict subgoal_prop_dict,
        structure::History history) {
    structure::SubgoalPropDict updated_subgoal_prop_dictionary;
    updated_subgoal_prop_dictionary = get_updated_subgoal_prop_dictionary(subgoal_prop_dict, history);
    return compute_subgoal_props_for_action(action, updated_subgoal_prop_dictionary);
}

structure::History add_to_history(
        std::tuple<int, int> state,
        structure::History history,
        structure::UnkDfaTransitions unk_dfa_transitions,
        int success) {
    structure::History updated_history = history;
    int subgoal = std::get<0>(state);
    for (auto unk_dfa_transition : unk_dfa_transitions) {
        for (int i = 0; i < unk_dfa_transition.size(); i++) {
            if (unk_dfa_transition[i] == 1) {
                updated_history.push_back(std::make_tuple(subgoal, i, success));
            }
        }
    }
    return updated_history;
}

std::vector<ActionPtr> get_actions_from_dict(
        structure::History history,
        structure::SubgoalPropDict subgoal_prop_dict,
        ActionDict action_dict,
        std::tuple<int, int> state) {
    auto all_actions = action_dict.find(state)->second;
    std::vector<ActionPtr> feasible_actions;
    for (auto action : all_actions) {
        auto [PS, RS, RE] = compute_updated_subgoal_props_with_history(
            action,
            subgoal_prop_dict,
            history);
        if (PS > 0) {
            feasible_actions.push_back(action);
        }
    }
    return feasible_actions;
}
