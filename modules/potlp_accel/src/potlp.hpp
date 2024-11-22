#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <tuple>
#ifndef _POTLP_UTILITY
#define _POTLP_UTILITY
#include "potlp_utility.hpp"
#endif


struct POTLPNode {
    std::tuple<int, int> state;
    std::shared_ptr<POTLPNode> parent;
    ActionPtr prev_action;
    double cost;
    bool goal_reached;
    std::vector<ActionPtr> actions;
    std::vector<ActionPtr> unexplored_actions;
    // In children, first index is failure and second index is success
    std::vector<std::shared_ptr<POTLPNode>> children;
    std::vector<double> action_values;
    std::vector<int> action_n;
    int action_n_total;
    structure::History history;
    TreeActionAndSubgoalProps *tree_data;

    POTLPNode(std::tuple<int, int> current_state,
              TreeActionAndSubgoalProps& tree_data,
              std::shared_ptr<POTLPNode> p = nullptr,
              ActionPtr previous_action = nullptr,
              double action_cost = 0.0) {
        state = current_state;
        this->tree_data = &tree_data;
        parent = p;
        prev_action = previous_action;
        if (parent == nullptr) {
            history = structure::History();
            // cost should be soemthing
            cost = action_cost;
        } else {
            history = p->history;
            cost = p->cost + action_cost;
        }
        // If the state has (n, 0), then 0 denotes the goal is reached
        int dfa_state = std::get<1>(current_state);
        if (dfa_state != 0) {
            actions = get_actions_from_dict(
                this->history,
                this->tree_data->subgoal_prop_dict,
                this->tree_data->action_dict,
                current_state);
            unexplored_actions = actions;
            action_n_total = 0;
            for (int i = 0; i < actions.size(); i++) {
                action_values.push_back(0.0);
                action_n.push_back(0);
                // initialize success and failure state as nullptr in children for size of
                children.push_back(nullptr);
                children.push_back(nullptr);
            }
            goal_reached = false;
        } else {
            goal_reached = true;
        }

        if (!goal_reached) {
            if (unexplored_actions.size() == 0) {
                // std::cout << "No unexplored actions" << std::endl;
                // just putting this here to end the traversal
                goal_reached = true;
            }
        }
        // print actions
        // std::cout << "Actions for current state: "
        //     << std::get<0>(current_state) << "," << std::get<1>(current_state) << std::endl;
        // ActionPtr action = unexplored_actions[0];
        // std::cout << "Action: " << std::get<0>(action->known_state) << "," << std::get<1>(action->known_state) << std::endl;
        // std::cout << "Action cost: " << action->known_space_cost << std::endl;
        // std::cout << "Action unk dfa state: " << action->unk_dfa_state << std::endl;
        // std::cout << "Unk DFA Transitions: " << std::endl;
        // for (auto transition: action->unk_dfa_transitions) {
        //     for (auto t: transition) {
        //         std::cout << t << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "Action node name path: " << std::endl;
        // for (auto node_name: action->node_name_path) {
        //     std::cout << std::get<0>(node_name) << " " << std::get<1>(node_name) << std::endl;
        // }
    }

    bool is_fully_explored() const {return (unexplored_actions.size() == 0);}

    bool is_terminal_state() const {
        return ((prev_action != nullptr) && prev_action->is_terminal) || goal_reached;
    }

    void clear_children() {
        for (auto const &child : children) {
            if (child != nullptr) {
                child->clear_children();
            }
        }
        children.clear();
    }

    double find_rollout_cost() {
        return cost;
    }

    ActionPtr potlp_best_action() {
    // find indices with maximum visits in root->action_n
        int best_action_index =  std::max_element(action_n.begin(), action_n.end()) - action_n.begin();
        return actions[best_action_index];
    }
};

inline double rollout(const std::shared_ptr<POTLPNode> &node) {
  return node->find_rollout_cost();
}

void backpropagate(std::shared_ptr<POTLPNode> node, double simulation_result) {
  /* Update the node and it's parent. We are updating node parents' properties
  because we find best action from a node rather than best node using uct */
  while (node->parent != nullptr) {
    int action_idx = find_action_index(node->prev_action, node->parent->actions);
    node->parent->action_n[action_idx] += 1;
    node->parent->action_n_total += 1;
    node->parent->action_values[action_idx] += simulation_result;
    node = node->parent;
  }
}

ActionPtr best_uct_action_potlp(std::shared_ptr<POTLPNode> node, const double &C = 50) {
    /*Pick the best action according to the UCB/UCT algorithm*/
    std::vector<double> Q;
    const std::vector<int> &n = node->action_n;

    for (int i = 0; i < n.size(); i++) {
        Q.push_back(node->action_values[i] / n[i] -
                    C * sqrt(log(node->action_n_total) / n[i]));
    }

    return node->actions[std::min_element(Q.begin(), Q.end()) - Q.begin()];
}

std::shared_ptr<POTLPNode> traverse(std::shared_ptr<POTLPNode> node) {
    ActionPtr action;
    std::mt19937 rng { std::random_device {} ()};
    std::shared_ptr<POTLPNode> current_node = node;
    int action_idx;
    std::shared_ptr<POTLPNode> new_node = nullptr;

    // If fully explored, pick one of the children
    while (current_node->is_fully_explored() && !current_node->is_terminal_state()) {
        action = best_uct_action_potlp(current_node);
        action_idx = find_action_index(action, current_node->actions);  // index where children should be stored
        // The action could lead to success or failure state with certain probability
        auto [PS, RS, RE] = compute_updated_subgoal_props_with_history(action,
                                            current_node->tree_data->subgoal_prop_dict,
                                            current_node->history);
        // Use Bernoulli's sampling to find if the action is successful
        std::bernoulli_distribution d(PS);
        bool success = d(rng);
        if (current_node->children[2 * action_idx + success] != nullptr) {
            current_node = current_node->children[action_idx * 2 + success];
        } else {
            std::tuple<int, int> state;
            double action_cost;
            if (success) {
                state = std::make_tuple(std::get<0>(action->known_state), action->unk_dfa_state);
                action_cost = action->known_space_cost + RS;
                // std::cout << "Success State: " << std::get<0>(state) << ", " << std::get<1>(state) << std::endl;
                // std::cout << "Known cost = " << action->known_space_cost << ", RS = " << RS << std::endl;
            } else {
                state = action->known_state;
                action_cost = action->known_space_cost + RE;
                // std::cout << "Failure State: " << std::get<0>(state) << ", " << std::get<1>(state) << std::endl;
                // std::cout << "Known cost = " << action->known_space_cost << ", RE = " << RE << std::endl;
            }
            std::shared_ptr<POTLPNode> new_node = std::make_shared<POTLPNode>(state,
                                                *current_node->tree_data, current_node, action, action_cost);
            // Add success/failure in the history
            new_node->history = add_to_history(state, new_node->history, action->unk_dfa_transitions, success);

            current_node->children[action_idx * 2 + success] = new_node;
            current_node = new_node;
        }
    }
    /* If the node is terminal state or goal state; return the node*/
    if (current_node->is_terminal_state())
        return current_node;

    /* If the node is not terminal */

    // 1). Pick an action from first index of the unexplored actions and remove it from unexplored actions*/
    action = current_node->unexplored_actions[0];
    current_node->unexplored_actions.erase(current_node->unexplored_actions.begin());
    // Find Ps for that action
    auto [PS, RS, RE] = compute_updated_subgoal_props_with_history(action,
                                current_node->tree_data->subgoal_prop_dict,
                                current_node->history);
    // Use Bernoulli's sampling to find if the action is successful
    std::bernoulli_distribution d(PS);
    bool success = d(rng);
    // 2.) Create a new child (new leaf)
    // if success, create success node else create failure node
    std::tuple<int, int> state;
    double action_cost = 0;
    if (success) {
        state = std::make_tuple(std::get<0>(action->known_state), action->unk_dfa_state);
        action_cost = action->known_space_cost + RS;
        // std::cout << "Success State: " << std::get<0>(state) << ", " << std::get<1>(state) << std::endl;
        // std::cout << "Known cost = " << action->known_space_cost << ", RS = " << RS << std::endl;
    } else {
        state = action->known_state;
        // print state
        action_cost = action->known_space_cost + RE;
        // std::cout << "Failure State: " << std::get<0>(state) << ", " << std::get<1>(state) << std::endl;
        // std::cout << "Known cost = " << action->known_space_cost << ", RE = " << RE << std::endl;
    }
    new_node = std::make_shared<POTLPNode>(state, *current_node->tree_data, current_node, action, action_cost);
    // Add success/failure in the history
    new_node->history = add_to_history(state, new_node->history, action->unk_dfa_transitions, success);

    // 3.) Add the child to the list of children
    action_idx = find_action_index(action, current_node->actions);  // index where children should be stored
    current_node->children[2*action_idx + success] = new_node;
    // 4.) return the child
    return new_node;
}



ActionPtr find_best_action(std::tuple<int, int> initial_state,
            ActionDict action_dict,
            structure::SubgoalPropDict subgoal_prop_dict,
            int num_iterations) {
        // Make root of the tree
        TreeActionAndSubgoalProps tree_data = TreeActionAndSubgoalProps(action_dict, subgoal_prop_dict);
        std::shared_ptr<POTLPNode> root = std::make_shared<POTLPNode>(initial_state, tree_data);
        // Loop through MCTS iterations
        for (int i = 0; i < num_iterations; i++) {
            // 1.) traverse the loop and get the leaf
            std::shared_ptr<POTLPNode> leaf = traverse(root);
            // 2.) get the rollout value
            double rollout_value = rollout(leaf);
            // std::cout << "Rollout value: " << rollout_value << std::endl;
            // 3.) backpropagate the value
            backpropagate(leaf, rollout_value);
        }

        // find the best action
        ActionPtr best_action = root->potlp_best_action();
        // return the best action
        root->clear_children();
        // std::cout << "children cleared" << std::endl;
        return best_action;
}
