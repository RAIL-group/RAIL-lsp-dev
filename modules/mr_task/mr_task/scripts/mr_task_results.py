import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mrlsp


def extract_costs(file_path):
    learned_cost = []
    optimistic_cost = []
    learnedgreedy_cost = []
    seed_costs = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(' : ')[1].strip())
            planner = parts[1].split(' : ')[1].strip()
            cost = float(parts[2].split(' : ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"learned": None, "optimistic": None, "learnedgreedy": None}
            seed_costs[seed][planner] = cost

    for seed in sorted(seed_costs.keys()):
        learned_cost.append(seed_costs[seed]["learned"])
        optimistic_cost.append(seed_costs[seed]["optimistic"])
        learnedgreedy_cost.append(seed_costs[seed]["learnedgreedy"])

    return learned_cost, optimistic_cost, learnedgreedy_cost, seed_costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/mr_task/mr_task_eval')
    parser.add_argument('--num_robots', type=int, default=2)
    parser.add_argument('--resolution', type=float, default=0.05)
    args = parser.parse_args()
    file_path = Path(args.save_dir) / f'log_{args.num_robots}.txt'

    learned_cost, optimistic_cost, learnedgreedy_cost, seed_costs = extract_costs(file_path)
    assert len(learned_cost) == len(optimistic_cost)
    assert len(learned_cost) == len(learnedgreedy_cost)
    mrlsp.utils.plotting.make_scatter_plot_with_box(optimistic_cost, learned_cost, xlabel='Optimistic', ylabel='Learned')
    image_name = Path(args.save_dir) / f'log_{args.num_robots}_scatter_learned_vs_optimistic.png'
    plt.savefig(image_name)
    image_name = Path(args.save_dir) / f'log_{args.num_robots}_scatter_learned_vs_optimistic.pdf'
    plt.savefig(image_name, dpi=1000)

    mrlsp.utils.plotting.make_scatter_plot_with_box(learnedgreedy_cost, learned_cost, xlabel='Learned Myopic', ylabel='Learned')
    image_name = Path(args.save_dir) / f'log_{args.num_robots}_scatter_learned_vs_learnedgreedy.pdf'
    plt.savefig(image_name, dpi=1000)
    image_name = Path(args.save_dir) / f'log_{args.num_robots}_scatter_learned_vs_learnedgreedy.png'
    plt.savefig(image_name)
