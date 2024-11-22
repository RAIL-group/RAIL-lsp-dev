
#ifndef _POTLP
#define _POTLP
#include "potlp.hpp"
#endif

#ifndef _TYPEDEF
#define _TYPEDEF
#include "typedef.h"
#endif

#include <tuple>
#include <map>
#include <vector>
#include <memory>


std::tuple<structure::SubgoalPropDict, structure::SubgoalPropDict>
    get_subgoal_prop_dictionary() {
        structure::SubgoalPropDict subgoal_prop_dict;
        structure::SubgoalPropDict target_subgoal_prop_dict;
        /* Subgoal_prop_dict tells us - for a given subgoal (100, 101, 102) - what is the Ps {1st index}, Rs {2nd index},
            and Re {3rd index} of items (list) beyond that subgoal is.
            For eg: Beyond subgoal 1, the Ps for finding item 1 is 0.2, item 2 is 0.3, and item 3 is 0.8,
                    and the associated Rs is 170, 120, 40 for item{1, 2, and 3} respectively.
                    Similarly Re is 40, 20 and 50 for item{1,2, and 3} respectively*/
        subgoal_prop_dict = {{100, {{0.2, 0.3, 0.8}, {170, 120, 40}, {40, 20, 50}}},
                            {101, {{0.3, 0.8, 0.4}, {180, 10, 200}, {50, 10, 20}}},
                            {102, {{0.4, 0.6, 0.8}, {190, 20, 80}, {60, 70, 50}}}};

        target_subgoal_prop_dict = {{100, {{1, 1, 0}, {170, 120, 40}, {40, 20, 50}}},
                                    {101, {{0.3, 0, 0.4}, {180, 10, 200}, {50, 10, 20}}},
                                    {102, {{0.4, 0.6, 1}, {190, 20, 80}, {60, 70, 50}}}};

        return std::make_tuple(subgoal_prop_dict, target_subgoal_prop_dict);
}

// Test functions that Greg wanted
bool test_update_subgoal_prop_dict() {
    structure::SubgoalPropDict subgoal_prop_dict;
    structure::SubgoalPropDict calculated_subgoal_prop_dict;
    structure::SubgoalPropDict target_subgoal_prop_dict;
    structure::History history;
    std::tie(subgoal_prop_dict, target_subgoal_prop_dict) = get_subgoal_prop_dictionary();
    history = {{100, 0, 1}, {100, 1, 1}, {100, 2, 0}, {101, 1, 0}, {102, 2, 1}};
    // The history tells: {subgoal, index of item, 1/0} where 1 means item is present and 0 means item is absent.
        // Subgoal 100 has item in 0th index.
        // Subgoal 100 has item in 1st index.
        // Subgoal 100 doesn't have item in 2nd index.
        // Subgoal 101 doesn't have item in 1st index.
        // Subgoal 102 has item in 2nd index.

    calculated_subgoal_prop_dict = get_updated_subgoal_prop_dictionary(subgoal_prop_dict, history);
    // Check if calculated_subgoal_prop_dict is equal to target_subgoal_prop_dict
    for (auto const& subgoal : calculated_subgoal_prop_dict) {
        for (int i = 0; i < subgoal.second.size(); i++) {
            for (int j = 0; j < subgoal.second[i].size(); j++) {
                if (subgoal.second[i][j] != target_subgoal_prop_dict[subgoal.first][i][j]) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_empty_history_subgoal_prop_dict_update() {
    structure::SubgoalPropDict subgoal_prop_dict;
    structure::SubgoalPropDict calculated_subgoal_prop_dict;
    structure::SubgoalPropDict target_subgoal_prop_dict;
    structure::History history = {};
    std::tie(subgoal_prop_dict, std::ignore) = get_subgoal_prop_dictionary();
    target_subgoal_prop_dict = subgoal_prop_dict;
    calculated_subgoal_prop_dict = get_updated_subgoal_prop_dictionary(subgoal_prop_dict, history);
    // Check if calculated_subgoal_prop_dict is equal to target_subgoal_prop_dict
    for (auto const& subgoal : calculated_subgoal_prop_dict) {
        for (int i = 0; i < subgoal.second.size(); i++) {
            for (int j = 0; j < subgoal.second[i].size(); j++) {
                if (subgoal.second[i][j] != target_subgoal_prop_dict[subgoal.first][i][j]) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_updated_ps_with_updated_properties() {
    structure::SubgoalPropDict subgoal_prop_dict, calculated_subgoal_prop_dict;
    structure::History history;
    std::tie(subgoal_prop_dict, std::ignore) = get_subgoal_prop_dictionary();
    history = {{100, 0, 1}, {100, 1, 1}, {100, 2, 0}, {101, 1, 0}, {102, 2, 1}};
    calculated_subgoal_prop_dict = get_updated_subgoal_prop_dictionary(subgoal_prop_dict, history);
    std::tuple<int, int> start_state = std::make_tuple(100, 1);
    std::tuple<int, int> known_state = std::make_tuple(101, 1);
    ActionPtr action = std::make_shared<Action>(start_state, known_state,
                                                1000,
                                                std::vector<std::tuple<int, int>>{start_state, known_state},
                                                0,
                                                std::vector<std::vector<int>>{{0, 0, 1}, {1, 1, 0}},
                                                false,
                                                1111);

    double target_ps_before_update = 0.544;
    double target_ps_after_update = 0.58;
    double calculated_ps_before_update = 0.0;
    double calculated_ps_after_update = 0.0;
    std::tie(calculated_ps_before_update, std::ignore, std::ignore) = compute_subgoal_props_for_action(action,
                                                                                            subgoal_prop_dict);
    std::tie(calculated_ps_after_update, std::ignore, std::ignore) = compute_subgoal_props_for_action(action,
                                                                                    calculated_subgoal_prop_dict);

    return ((std::abs(calculated_ps_before_update - target_ps_before_update) < 0.01 ) &&
           (std::abs(calculated_ps_after_update == target_ps_after_update) < 0.01));
}

bool test_add_to_history(structure::History history,
                        structure::UnkDfaTransitions unk_dfa_transitions,
                        structure::History target_history,
                        int success) {
    std::tuple<int, int> state = std::make_tuple(100, 1);
    structure::History calculated_history = add_to_history(state, history, unk_dfa_transitions, success);
    // print calculated history
    std::cout << "Calculated history: ";
    for (auto const& item : calculated_history) {
        std::cout << "{" << std::get<0>(item) << ", " << std::get<1>(item) << ", "
            << std::get<2>(item) << "}, "<< std::endl;
    }
    // Check if calculated_history is equal to target_history
    for (int i = 0; i < calculated_history.size(); i++) {
        if (calculated_history[i] != target_history[i]) {
            return false;
        }
    }
    return true;
}

std::tuple<double, double, double> test_get_ps_rs_re_with_history(structure::SubgoalPropDict subgoal_prop_dict,
                                        structure::History history,
                                        ActionPtr action) {
    structure::SubgoalPropDict calculated_subgoal_prop_dict;
    calculated_subgoal_prop_dict = get_updated_subgoal_prop_dictionary(subgoal_prop_dict, history);
    auto[ps, rs, re] = compute_subgoal_props_for_action(action, calculated_subgoal_prop_dict);

    return std::make_tuple(ps, rs, re);
}
