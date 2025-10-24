import numpy as np
EXPLORATION_C = 100


def compute_lbcost_wavg(true_costs, lb_costs, chosen_planner_idx, prob_shortcut=0):
    """Computes weighted average of optimistic and simply-connected lower bound cost."""
    optimistic_lb = lb_costs[:, 0]
    simply_connected_lb = lb_costs[:, 1]
    # Use simply connected lb values if optimistic lb values are infinity
    optimistic_lb[np.isinf(optimistic_lb)] = simply_connected_lb[np.isinf(optimistic_lb)]
    # Compute weighted average
    wavg = prob_shortcut * optimistic_lb + (1 - prob_shortcut) * simply_connected_lb
    # For chosen planner, true cost is returned instead
    wavg[chosen_planner_idx] = true_costs[chosen_planner_idx]

    return wavg


def get_lb_selection(weighted_costs, tot_cost_per_planner, num_selection_per_planner, min_idx, c=EXPLORATION_C):
    costs_try = np.zeros_like(weighted_costs)
    costs_try[min_idx] = weighted_costs[min_idx]
    costs_sim = weighted_costs.copy()
    costs_sim[min_idx] = 0
    tot_cost_per_planner[0] += costs_try
    tot_cost_per_planner[1] += costs_sim
    num_selection_per_planner[0, min_idx] += 1
    num_selection_per_planner[1] += 1
    num_selection_per_planner[1, min_idx] -= 1

    num_trials = num_selection_per_planner[0].sum()
    # if num_trials < 1:
    #     num_trials = 1
    # Compute mean costs for each planner
    # mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner + 0.001)
    mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner)
    mean_cost_per_planner[np.isnan(mean_cost_per_planner)] = 0
    # Compute weighted average cost based on true and simulated lb costs
    cost_wavg = (num_selection_per_planner[0] * mean_cost_per_planner[0] +
                 num_selection_per_planner[1] * mean_cost_per_planner[1]) / num_trials
    cost_wavg[np.isnan(cost_wavg)] = 0
    # Compute exploration magnitude of UCB
    bandit_exploration_magnitude = np.sqrt(np.log(num_trials) /
                                           (num_selection_per_planner[0]))
    # Compute UCB bandit cost as usual
    bandit_cost = (mean_cost_per_planner[0] - c * bandit_exploration_magnitude)
    # Compute final cost used for selection (Const-UCB cost)
    our_cost = np.maximum(bandit_cost, cost_wavg)
    # Compute the planner index with minimum Const-UCB cost
    min_idx = np.argmin(our_cost)

    return tot_cost_per_planner, num_selection_per_planner, min_idx


def get_ucb_selection(weighted_costs, tot_cost_per_planner, num_selection_per_planner, min_idx, c=EXPLORATION_C):
    costs_try = np.zeros_like(weighted_costs)
    costs_try[min_idx] = weighted_costs[min_idx]
    # costs_sim = weighted_costs.copy()
    # costs_sim[min_idx] = 0
    tot_cost_per_planner[0] += costs_try
    # tot_cost_per_planner[1] += costs_sim
    num_selection_per_planner[0, min_idx] += 1
    # num_selection_per_planner[1] += 1
    # num_selection_per_planner[1, min_idx] -= 1

    num_trials = num_selection_per_planner[0].sum()
    # if num_trials < 1:
    #     num_trials = 1
    # Compute mean costs for each planner
    # mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner + 0.001)
    mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner)
    mean_cost_per_planner[np.isnan(mean_cost_per_planner)] = 0
    # Compute weighted average cost based on true and simulated lb costs
    # cost_wavg = (num_selection_per_planner[0] * mean_cost_per_planner[0] +
    #              num_selection_per_planner[1] * mean_cost_per_planner[1]) / num_trials
    # cost_wavg[np.isnan(cost_wavg)] = 0
    # Compute exploration magnitude of UCB
    bandit_exploration_magnitude = np.sqrt(np.log(num_trials) /
                                           (num_selection_per_planner[0]))
    # Compute UCB bandit cost as usual
    bandit_cost = (mean_cost_per_planner[0] - c * bandit_exploration_magnitude)
    # Compute final cost used for selection (Const-UCB cost)
    # our_cost = np.maximum(bandit_cost, cost_wavg)
    # Compute the planner index with minimum Const-UCB cost
    min_idx = np.argmin(bandit_cost)

    return tot_cost_per_planner, num_selection_per_planner, min_idx
