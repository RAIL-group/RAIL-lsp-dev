import lsp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

NUM_TRIALS = 100
NUM_SAMPLING = 500


def compute_ucb_bandit_cost(env, planners, c=100, random_seed=42):
    """Computes UCB bandit cost when deployed in a given environment.
    Cost of a planner for each trial is read from file based on
    chosen planner, all available planers, environment name and map seed.

    Returns cumulative costs over trials, selection rates and chosen planner indices.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(random_seed)

    # Shuffle seeds to randomize the order of maps
    np.random.shuffle(seeds)
    tot_cost_per_planner = np.zeros(len(planners))
    num_selection_per_planner = np.zeros(len(planners))

    chosen_indices = []
    all_costs = []
    for _ in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        num_trials = num_selection_per_planner.sum()

        mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner)
        mean_cost_per_planner[np.isnan(mean_cost_per_planner)] = 0
        # Compute the planner index with minimum UCB cost
        ucb_cost = mean_cost_per_planner - c * np.sqrt(np.log(num_trials)
                                                       / ((num_selection_per_planner)))
        ucb_cost[np.isnan(ucb_cost)] = 0
        ucb_cost[np.isinf(ucb_cost)] = 0
        min_value = np.min(ucb_cost)
        min_indices = np.where(ucb_cost == min_value)[0]
        min_idx = np.random.choice(min_indices)
        # Update the chosen planner
        chosen_planner = planners[min_idx]
        chosen_indices.append(min_idx)

        # Get cost of chosen planner for current trial
        cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{seed}.txt'
        cost = np.loadtxt(cost_file)[min_idx]

        # Update costs and times selected
        all_costs.append(cost)
        tot_cost_per_planner[min_idx] += cost
        num_selection_per_planner[min_idx] += 1

    return (np.cumsum(all_costs) / (np.arange(NUM_TRIALS) + 1),
            num_selection_per_planner / NUM_TRIALS, chosen_indices)


def compute_base_planner_costs(env, planners, seed=42):
    """Computes cumulative costs of each planner over trials in an environment
    without performing any selection.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(seed)
    np.random.shuffle(seeds)
    costs_per_planner = np.zeros((NUM_TRIALS, len(planners)))
    for i in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        for j, chosen_planner in enumerate(planners):
            cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{seed}.txt'
            cost = np.loadtxt(cost_file)
            costs_per_planner[i, j] = cost[j]

    return np.cumsum(costs_per_planner, axis=0) / (np.arange(NUM_TRIALS).reshape(-1, 1) + 1)


def compute_lbcost_wavg(env, chosen_planner, env_seed, prob_shortcut=0):
    """Computes weighted average of optimistic and simply-connected lower bound cost
    for a planner in a given map seed based on likelihood of finding a shorter path to goal
    in the environment. For chosen planner, true cost is computed.
    """
    lb_costs_file = Path(args.save_dir) / f'lbc_{chosen_planner}_all_{all_planners}_{env}_{env_seed}.txt'
    cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{env_seed}.txt'
    true_costs = np.loadtxt(cost_file)
    lb_costs = np.loadtxt(lb_costs_file)
    optimistic_lb = lb_costs[:, 0]
    simply_connected_lb = lb_costs[:, 1]
    # Use simply connected lb values if optimistic lb values are infinity
    optimistic_lb[np.isinf(optimistic_lb)] = simply_connected_lb[np.isinf(optimistic_lb)]
    # Compute weighted average
    wavg = prob_shortcut * optimistic_lb + (1 - prob_shortcut) * simply_connected_lb
    chosen_planner_idx = planners.index(chosen_planner)
    # For chosen planner, true cost is returned instead
    wavg[chosen_planner_idx] = true_costs[chosen_planner_idx]

    return wavg


def compute_lb_selection_cost(env, planners, c=100, prob_shortcut=0, random_seed=42):
    """Computes Const-UCB (ours) cost when deployed in a given environment.
    The function 'compute_lbcost_wavg' is used to get true cost for chosen planner and
    weighted lowerbound costs for other planners which are used to select among planners
    using modified UCB bandit appoach.

    Returns cumulative costs over trials, selection rates and chosen planner indices.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(random_seed)
    np.random.shuffle(seeds)
    # Store true cost (first row) and simulated lb costs (second row)
    tot_cost_per_planner = np.zeros((2, len(planners)))
    num_selection_per_planner = np.zeros((2, len(planners)))

    chosen_indices = []
    all_costs = []
    for i in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        num_trials = num_selection_per_planner[0].sum()
        # Compute mean costs for each planner
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
        bandit_cost[np.isnan(bandit_cost)] = 0
        bandit_cost[np.isinf(bandit_cost)] = 0
        our_cost = np.maximum(bandit_cost, cost_wavg)
        # Compute the planner index with minimum Const-UCB cost
        min_value = np.min(our_cost)
        min_indices = np.where(our_cost == min_value)[0]
        min_idx = np.random.choice(min_indices)
        # min_idx = np.argmin(our_cost)
        # Update the chosen planner
        chosen_planner = planners[min_idx]
        chosen_indices.append(min_idx)

        # Get true cost (for chosen planner) and simulated lb costs (for other planners) for current trial
        costs = compute_lbcost_wavg(env, chosen_planner, seed, prob_shortcut=prob_shortcut)
        # if i == 0:
        #     print(costs)

        # Make updates to planner costs and times selected
        all_costs.append(costs[min_idx])
        costs_try = np.zeros_like(costs)
        costs_try[min_idx] = costs[min_idx]
        costs_sim = costs.copy()
        costs_sim[min_idx] = 0
        tot_cost_per_planner[0] += costs_try
        tot_cost_per_planner[1] += costs_sim
        num_selection_per_planner[0, min_idx] += 1
        num_selection_per_planner[1] += 1
        num_selection_per_planner[1, min_idx] -= 1

    return (np.cumsum(all_costs) / (np.arange(NUM_TRIALS) + 1),
            num_selection_per_planner[0] / NUM_TRIALS, chosen_indices)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--start_seeds', type=int, nargs='+', default=[1000])
    parser.add_argument('--num_seeds', type=int, default=150)
    args = parser.parse_args()

    planners = ['prompttrivial',
                'lspgptpromptone', 'lspgptprompttwo', 'lspgptpromptthree',
                'lspgeminipromptone', 'lspgeminiprompttwo', 'lspgeminipromptthree',
                'fullgptpromptone',
                'fullgeminipromptone']
    planner_names = ['OPTIMISTIC+MODEL',
                     'LLM+MODEL/P-CONTEXT-A/GPT-4o', 'LLM+MODEL/P-CONTEXT-B/GPT-4o', 'LLM+MODEL/P-MINIMAL/GPT-4o',
                     'LLM+MODEL/P-CONTEXT-A/Gemini', 'LLM+MODEL/P-CONTEXT-B/Gemini', 'LLM+MODEL/P-MINIMAL/Gemini',
                     'LLM-DIRECT/P-DIRECT/GPT-4o',
                     'LLM-DIRECT/P-DIRECT/Gemini']
    planner_plot_order = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    our_planner_gpt_idx = [1, 2, 3]
    our_planner_gemini_idx = [4, 5, 6]
    direct_planner_gpt_idx = 7
    direct_planner_gemini_idx = 8
    optim_planner_idx = 0

    # planner_colors = ['brown', 'green', 'gray', 'darkorange']
    envs = ['apartment']
    env_names = ['Apartment']
    env_seeds = {'apartment': (args.start_seeds[0], args.start_seeds[0] + args.num_seeds)}

    all_planners = '_'.join(planners)

    trials_to_print = np.array([20, 50, 100]) - 1
    trial_markers = ['^', 'd', 's']
    trial_marker_size = 9
    ucb_color = 'tab:orange'
    best_policy_color = 'tab:green'
    fill_alpha = 0.08
    xticks = list(range(0, NUM_TRIALS + 1, 20))
    xticks[0] = 1

    env_planner_costs = {}

    print('------------------------Base Planner Results----------------------------')
    for i, env in enumerate(envs):
        print(f'-------------------------------{env_names[i]}-------------------------------')
        dat = [compute_base_planner_costs(env, planners, seed=seed) for seed in range(NUM_SAMPLING)]
        dat = np.array(dat)
        planner_costs = []
        for j, planner in enumerate(planners):
            all_runs = dat[:, :, j]
            planner_avg_cost = np.mean(all_runs, axis=0)[-1]
            planner_costs.append(planner_avg_cost)
            print(f'Incurred Cost [{planner_names[j]:<20}]: {planner_avg_cost:.2f}')
        env_planner_costs[i] = np.array(planner_costs)

    probs = [0.0]
    tags = [r'$C^{lb}$']
    y_labels = ['LB Cost']
    colors = ['tab:blue']

    fig_costs, axs_costs = plt.subplots(1, len(envs), figsize=(6, 3))
    fig_regret, axs_regret = plt.subplots(1, len(envs), figsize=(6, 2.1))
    fig_rates, axs_rates = plt.subplots(len(probs), len(envs), figsize=(6, 3.3))
    axs_costs = [axs_costs] if len(envs) == 1 else axs_costs
    axs_regret = [axs_regret] if len(envs) == 1 else axs_regret
    axs_rates = [axs_rates] if len(envs) == 1 else axs_rates
    axs_rates = [axs_rates] if len(probs) == 1 else axs_rates

    fig_costs.subplots_adjust(wspace=0.3, top=0.95, bottom=0.03, left=0.13, right=0.985)
    fig_regret.subplots_adjust(wspace=0.3, top=1, bottom=0.276, left=0.13, right=0.985)
    fig_rates.subplots_adjust(wspace=0.3, top=1, bottom=0.157, left=0.41, right=0.985)

    for i, env in enumerate(envs):
        print(f'\n-------------------------------{env_names[i]}-------------------------------')
        print('----------------------UCB-Bandit Results-----------------------')
        dat = [compute_ucb_bandit_cost(env, planners, random_seed=seed) for seed in range(NUM_SAMPLING)]
        all_runs, pull_rates, all_chosen_indx = zip(*dat)
        avg_costs_ucb = np.mean(all_runs, axis=0)
        p10_costs_ucb = np.percentile(all_runs, 10, axis=0)
        p90_costs_ucb = np.percentile(all_runs, 90, axis=0)

        best_asymp_cost = min(env_planner_costs[i])

        # UCB Cost
        plt.sca(axs_costs[i])
        axs_costs[i].spines['top'].set_visible(False)
        axs_costs[i].spines['right'].set_visible(False)

        plt.plot(range(1, NUM_TRIALS + 1),
                 avg_costs_ucb,
                 color=ucb_color,
                 linewidth=3,
                 label='UCB Selection')
        # plt.fill_between(range(1, NUM_TRIALS + 1),
        #                  p10_costs_ucb,
        #                  p90_costs_ucb,
        #                  alpha=fill_alpha,
        #                  color=ucb_color)

        for m, trial in enumerate(trials_to_print):
            plt.plot(trial + 1,
                     avg_costs_ucb[trial],
                     marker=trial_markers[m],
                     markersize=trial_marker_size,
                     color=ucb_color)
            print(f'Trial {trial + 1} Cost: {avg_costs_ucb[trial]:.2f}')

        plt.xticks(xticks, fontsize='x-large')
        plt.xlim([1, NUM_TRIALS + 2])
        plt.gca().set_xticklabels([])
        plt.ylabel('Avg. Navigation Cost', fontsize='x-large')
        plt.ylim([180, 250])
        plt.yticks(range(180, 241, 20), fontsize='x-large')

        regrets_ucb = np.cumsum(all_runs - best_asymp_cost, axis=1)
        avg_regrets_ucb = regrets_ucb.mean(0)

        # UCB Pull Rate
        print('Percentage of times each planner was selected for UCB:')
        selection_counts_ucb = [[] for _ in planners]
        for chosen_indices in all_chosen_indx:
            for planner_idx, count in zip(*np.unique(chosen_indices, return_counts=True)):
                selection_counts_ucb[planner_idx].append(count)
        mean_counts_ucb = np.array([np.mean(counts) for counts in selection_counts_ucb])
        rates_ucb = mean_counts_ucb / NUM_TRIALS * 100
        print(planner_names)
        print(rates_ucb)

        # UCB Regret
        plt.sca(axs_regret[i])
        axs_regret[i].spines['top'].set_visible(False)
        axs_regret[i].spines['right'].set_visible(False)
        plt.plot(range(1, NUM_TRIALS + 1),
                 avg_regrets_ucb,
                 color=ucb_color,
                 linewidth=3,
                 label='UCB Selection')
        # plt.fill_between(range(1, NUM_TRIALS + 1),
        #                  np.percentile(regrets_ucb, 10, axis=0),
        #                  np.percentile(regrets_ucb, 90, axis=0),
        #                  alpha=fill_alpha,
        #                  color=ucb_color)

        for m, trial in enumerate(trials_to_print):
            plt.plot(trial + 1,
                     avg_regrets_ucb[trial],
                     marker=trial_markers[m],
                     markersize=trial_marker_size,
                     color=ucb_color,
                     linewidth=3,
                     fillstyle='none')
            print(f'Trial {trial + 1} Regret: {avg_regrets_ucb[trial]:.1f}')

        plt.xlabel(f'Num of Trials ({r"$k$"})', fontsize='x-large')
        plt.xticks(xticks, fontsize='x-large')
        plt.xlim([1, NUM_TRIALS + 2])
        plt.ylabel('Cumulative Regret', fontsize='x-large')
        plt.ylim([0, 3000])
        plt.yticks(range(0, 3000, 1000), fontsize='x-large')

        print('----------------------Replay Selection Results--------------------------')
        for k, p_short in enumerate(probs):
            print(f'--------------------------{p_short=}--------------------------')
            dat = [compute_lb_selection_cost(env, planners, prob_shortcut=p_short, random_seed=seed)
                   for seed in range(NUM_SAMPLING)]
            all_runs, pull_rates, all_chosen_indx = zip(*dat)
            avg_costs_const_ucb = np.mean(all_runs, axis=0)
            p10_costs_const_ucb = np.percentile(all_runs, 10, axis=0)
            p90_costs_const_ucb = np.percentile(all_runs, 90, axis=0)

            # Const-UCB Cost
            plt.sca(axs_costs[i])
            axs_costs[i].spines['top'].set_visible(False)
            axs_costs[i].spines['right'].set_visible(False)

            plt.plot(range(1, NUM_TRIALS + 1),
                     avg_costs_const_ucb,
                     color=colors[k],
                     linewidth=3,
                     label='Replay Selection (ours)')
            # plt.fill_between(range(1, NUM_TRIALS + 1),
            #                  p10_costs_const_ucb,
            #                  p90_costs_const_ucb,
            #                  alpha=fill_alpha,
            #                  color=colors[k])

            plt.sca(axs_costs[i])
            for m, trial in enumerate(trials_to_print):
                plt.plot(trial + 1, avg_costs_const_ucb[trial],
                         marker=trial_markers[m],
                         markersize=trial_marker_size,
                         color=colors[k])
                print(f'Trial {trial + 1} Cost: {avg_costs_const_ucb[trial]:.2f}')

            regrets_const_ucb = np.cumsum(all_runs - best_asymp_cost, axis=1)
            avg_regrets_const_ucb = regrets_const_ucb.mean(0)

            # Const-UCB Pull Rate
            print('Number of times each planner was selected for Const-UCB:')
            selection_counts_const_ucb = [[] for _ in planners]
            for chosen_indices in all_chosen_indx:
                for planner_idx, count in zip(*np.unique(chosen_indices, return_counts=True)):
                    selection_counts_const_ucb[planner_idx].append(count)
            mean_counts_const_ucb = np.array([np.mean(counts) for counts in selection_counts_const_ucb])
            rates_const_ucb = mean_counts_const_ucb / NUM_TRIALS * 100
            print(planner_names)
            print(rates_const_ucb)

            # Const-UCB Regret
            plt.sca(axs_regret[i])
            plt.plot(range(1, NUM_TRIALS + 1),
                     avg_regrets_const_ucb,
                     color=colors[k],
                     linewidth=3,
                     label='Replay Selection (ours)')
            # plt.fill_between(range(1, NUM_TRIALS + 1),
            #                  np.percentile(regrets_const_ucb, 10, axis=0),
            #                  np.percentile(regrets_const_ucb, 90, axis=0),
            #                  alpha=fill_alpha,
            #                  color=colors[k])

            for m, trial in enumerate(trials_to_print):
                plt.plot(trial + 1,
                         avg_regrets_const_ucb[trial],
                         marker=trial_markers[m],
                         markersize=trial_marker_size,
                         color=colors[k],
                         linewidth=3,
                         fillstyle='none')
                print(f'Trial {trial + 1} Regret: {avg_regrets_const_ucb[trial]:.1f}')

            # Selection rates bar plot for each planner
            plt.sca(axs_rates[i][k])
            axs_rates[i][k].spines['top'].set_visible(False)
            axs_rates[i][k].spines['right'].set_visible(False)
            planner_names_sorted = [planner_names[j] for j in planner_plot_order]
            rates_ucb_sorted = rates_ucb[planner_plot_order]
            rates_const_ucb_sorted = rates_const_ucb[planner_plot_order]
            x_pos = np.arange(len(planner_names_sorted))
            plt.barh(x_pos - 0.2, rates_ucb_sorted, 0.4, color=ucb_color, label='UCB Selection')
            plt.barh(x_pos + 0.2, rates_const_ucb_sorted, 0.4, color=colors[k], label='Replay Selection (ours)')
            plt.xticks(range(0, 51, 10), fontsize='x-large')
            plt.yticks(x_pos, planner_names_sorted, fontsize='medium')
            plt.xlabel('Selection Rate (%)', fontsize='x-large')
            plt.xlim([0, 50])
            plt.gca().invert_yaxis()
            plt.legend(fontsize='large')

        plt.sca(axs_regret[i])
        plt.hlines(y=0, xmin=1, xmax=NUM_TRIALS,
                   colors=best_policy_color, linestyles='--', linewidth=3, label='Best Performance (oracle)')
        plt.sca(axs_costs[i])
        plt.hlines(y=best_asymp_cost, xmin=1, xmax=NUM_TRIALS,
                   colors=best_policy_color, linestyles='--', linewidth=3, label='Best Performance (oracle)')
        plt.legend(fontsize='large')

        ucb_cost_final = avg_costs_ucb[-1]
        our_cost_final = avg_costs_const_ucb[-1]
        percent_improve_ucb_cost = (ucb_cost_final - our_cost_final) / ucb_cost_final * 100

        print('--------------------------Planner Comparison--------------------------')
        p_names = np.array(planner_names)
        print('--------------------------GPT-4--------------------------')
        our_planner_gpt_costs = env_planner_costs[i][our_planner_gpt_idx]
        our_best_planner_gpt = np.argmin(our_planner_gpt_costs)
        our_best_planner_gpt_cost = our_planner_gpt_costs[our_best_planner_gpt]
        print(f'Our Best Planner GPT: {p_names[our_planner_gpt_idx][our_best_planner_gpt]}, Cost: {our_best_planner_gpt_cost:.2f}')
        our_worst_planner_gpt = np.argmax(our_planner_gpt_costs)
        our_worst_planner_gpt_cost = our_planner_gpt_costs[our_worst_planner_gpt]
        print(f'Our Worst Planner GPT: {p_names[our_planner_gpt_idx][our_worst_planner_gpt]}, Cost: {our_worst_planner_gpt_cost:.2f}')
        print('----------------------------------------------------')
        directllm_gpt_cost = env_planner_costs[i][direct_planner_gpt_idx]
        optimistic_cost = env_planner_costs[i][optim_planner_idx]
        percent_improve_gpt_our_best_vs_directllm = (directllm_gpt_cost - our_best_planner_gpt_cost) / directllm_gpt_cost * 100
        print(f'Improvement GPT: Our Best Planner wrt Full-LLM Planner: {percent_improve_gpt_our_best_vs_directllm:.1f}%')
        percent_improve_gpt_our_worst_vs_directllm = (directllm_gpt_cost - our_worst_planner_gpt_cost) / directllm_gpt_cost * 100
        print(f'Improvement GPT: Our Worst Planner wrt Full-LLM Planner: {percent_improve_gpt_our_worst_vs_directllm:.1f}%')
        percent_improve_gpt_our_best_vs_optimistic = (optimistic_cost - our_best_planner_gpt_cost) / optimistic_cost * 100
        print(f'Improvement GPT: Our Best Planner wrt Optimistic Planner: {percent_improve_gpt_our_best_vs_optimistic:.1f}%')
        percent_improve_gpt_our_worst_vs_optimistic = (optimistic_cost - our_worst_planner_gpt_cost) / optimistic_cost * 100
        print(f'Improvement GPT: Our Worst Planner wrt Optimistic Planner: {percent_improve_gpt_our_worst_vs_optimistic:.1f}%')
        print('----------------------------------------------------')

        print('--------------------------Gemini--------------------------')
        our_planner_gemini_costs = env_planner_costs[i][our_planner_gemini_idx]
        our_best_planner_gemini = np.argmin(our_planner_gemini_costs)
        our_best_planner_gemini_cost = our_planner_gemini_costs[our_best_planner_gemini]
        print(f'Our Best Planner Gemini: {p_names[our_planner_gemini_idx][our_best_planner_gemini]}, Cost: {our_best_planner_gemini_cost:.2f}')
        our_worst_planner_gemini = np.argmax(our_planner_gemini_costs)
        our_worst_planner_gemini_cost = our_planner_gemini_costs[our_worst_planner_gemini]
        print(f'Our Worst Planner Gemini: {p_names[our_planner_gemini_idx][our_worst_planner_gemini]}, Cost: {our_worst_planner_gemini_cost:.2f}')
        print('----------------------------------------------------')
        directllm_gemini_cost = env_planner_costs[i][direct_planner_gemini_idx]
        percent_improve_gemini_our_best_vs_directllm = (directllm_gemini_cost - our_best_planner_gemini_cost) / directllm_gemini_cost * 100
        print(f'Improvement Gemini: Our Best Planner wrt Full-LLM Planner: {percent_improve_gemini_our_best_vs_directllm:.1f}%')
        percent_improve_gemini_our_worst_vs_directllm = (directllm_gemini_cost - our_worst_planner_gemini_cost) / directllm_gemini_cost * 100
        print(f'Improvement Gemini: Our Worst Planner wrt Full-LLM Planner: {percent_improve_gemini_our_worst_vs_directllm:.1f}%')
        percent_improve_gemini_our_best_vs_optimistic = (optimistic_cost - our_best_planner_gemini_cost) / optimistic_cost * 100
        print(f'Improvement Gemini: Our Best Planner wrt Optimistic Planner: {percent_improve_gemini_our_best_vs_optimistic:.1f}%')
        percent_improve_gemini_our_worst_vs_optimistic = (optimistic_cost - our_worst_planner_gemini_cost) / optimistic_cost * 100
        print(f'Improvement Gemini: Our Worst Planner wrt Optimistic Planner: {percent_improve_gemini_our_worst_vs_optimistic:.1f}%')
        print('----------------------------------------------------')

        # print('--------------------------Optimistic--------------------------')
        # our_planner_gemini_costs = env_planner_costs[i][our_planner_gemini_idx]
        # our_best_planner_gemini = np.argmin(our_planner_gemini_costs)
        # our_best_planner_gemini_cost = our_planner_gemini_costs[our_best_planner_gemini]
        # print(f'Our Best Planner Gemini: {p_names[our_planner_gemini_idx][our_best_planner_gemini]}, Cost: {our_best_planner_gemini_cost:.2f}')
        # our_worst_planner_gemini = np.argmax(our_planner_gemini_costs)
        # our_worst_planner_gemini_cost = our_planner_gemini_costs[our_worst_planner_gemini]
        # print(f'Our Worst Planner Gemini: {p_names[our_planner_gemini_idx][our_worst_planner_gemini]}, Cost: {our_worst_planner_gemini_cost:.2f}')
        # print('----------------------------------------------------')
        # directllm_gemini_cost = env_planner_costs[i][direct_planner_gemini_idx]
        # percent_improve_gemini_our_best_vs_directllm = (directllm_gemini_cost - our_best_planner_gemini_cost) / directllm_gemini_cost * 100
        # print(f'Improvement Gemini: Our Best Planner wrt Full-LLM Planner: {percent_improve_gemini_our_best_vs_directllm:.1f}%')
        # percent_improve_gemini_our_worst_vs_directllm = (directllm_gemini_cost - our_worst_planner_gemini_cost) / directllm_gemini_cost * 100
        # print(f'Improvement Gemini: Our Worst Planner wrt Full-LLM Planner: {percent_improve_gemini_our_worst_vs_directllm:.1f}%')
        # print('----------------------------------------------------')

        print('--------------------------Final Selection Results--------------------------')
        print('Average Cost')
        print(f'UCB: {ucb_cost_final:.2f}, Ours: {our_cost_final:.2f}, '
              f'Improvement: {percent_improve_ucb_cost:.1f}%')

        ucb_regret_final = avg_regrets_ucb[-1]
        our_regret_final = avg_regrets_const_ucb[-1]
        percent_improve_ucb_regret = (ucb_regret_final - our_regret_final) / ucb_regret_final * 100

        print('Cumulative Regret')
        print(f'UCB: {ucb_regret_final:.2f}, Ours: {our_regret_final:.2f}, '
              f'Improvement: {percent_improve_ucb_regret:.1f}%')
        print('--------------------------------------------------------------------')

    results_dir = Path(args.save_dir) / 'results'
    results_dir.mkdir(exist_ok=True)
    fig_costs.savefig(results_dir / 'results_costs.png')
    fig_regret.savefig(results_dir / 'results_regret.png')
    fig_rates.savefig(results_dir / 'results_rates.png')
    fig_costs.savefig(results_dir / 'results_costs.pdf')
    fig_regret.savefig(results_dir / 'results_regret.pdf')
    fig_rates.savefig(results_dir / 'results_rates.pdf')
    print(f'Results plots saved in {results_dir} as '
          f'results_costs.png, results_regret.png and results_rates.png')
    # plt.show()
