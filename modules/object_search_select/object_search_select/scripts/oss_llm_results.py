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
RESOLUTION = 1
EXPLORATION_C = 100


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
            cost_file = (Path(args.save_dir) /
                         f'target_plcy_{chosen_planner}_envrnmnt_{env}_{seed}.txt')
            cost = np.loadtxt(cost_file) / RESOLUTION
            costs_per_planner[i, j] = cost

    return np.cumsum(costs_per_planner, axis=0) / (np.arange(NUM_TRIALS).reshape(-1, 1) + 1)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--start_seeds', type=int, nargs='+', default=[1000])
    parser.add_argument('--num_seeds', type=int, default=150)
    args = parser.parse_args()

    results_dir = Path(args.save_dir) / 'results'
    results_dir.mkdir(exist_ok=True)

    planners = [
        # 'lspllamaprompta', 'lspllamapromptb', 'lspllamapromptminimal',
        # 'fullllamapromptdirect',
        'lspgptossprompta', 'lspgptosspromptb', 'lspgptosspromptminimal',
        'fullgptosspromptdirect'
    ]
    planner_names = [
        # 'LLM+MODEL/P-CONTEXT-A/LLaMa3.2', 'LLM+MODEL/P-CONTEXT-B/LLaMa3.2', 'LLM+MODEL/P-MINIMAL/LLaMa3.2',
        # 'LLM-DIRECT/P-DIRECT/LLaMa3.2',
        'LLM+MODEL/P-CONTEXT-A/GPT-OSS', 'LLM+MODEL/P-CONTEXT-B/GPT-OSS', 'LLM+MODEL/P-MINIMAL/GPT-OSS',
        'LLM-DIRECT/P-DIRECT/GPT-OSS'
    ]
    planner_plot_order = [0, 1, 2, 3]
    our_planner_gpt_idx = [1, 2, 3]
    our_planner_gemini_idx = [4, 5, 6]
    direct_planner_gpt_idx = 7
    direct_planner_gemini_idx = 8
    optim_planner_idx = 0

    envs = ['apartment']
    env_names = ['Apartment']
    env_seeds = {'apartment': (args.start_seeds[0], args.start_seeds[0] + args.num_seeds)}

    all_planners = '_'.join(planners)

    env_planner_costs = {}
    env_planner_costs_all_seeds = {}
    print('------------------------Base Planner Results----------------------------')
    for i, env in enumerate(envs):
        print(f'-------------------------------{env_names[i]}-------------------------------')
        dat = [compute_base_planner_costs(env, planners, seed=seed) for seed in range(NUM_SAMPLING)]
        dat = np.array(dat)
        planner_costs = []
        planner_costs_all_seeds = []
        for j, planner in enumerate(planners):
            all_runs = dat[:, :, j]
            planner_avg_cost = np.mean(all_runs, axis=0)[-1]
            planner_costs_all_seeds.append(all_runs[:, -1])
            planner_std_cost = np.std(all_runs, axis=0)[-1]
            planner_costs.append(planner_avg_cost)
            print(f'Incurred Cost [{planner_names[j]:<20}]: {planner_avg_cost:.2f}')
        env_planner_costs[i] = np.array(planner_costs)
        env_planner_costs_all_seeds[i] = np.array(planner_costs_all_seeds)
