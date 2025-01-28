import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mrlsp


def extract_costs(file_path):
    learned_cost = []
    optimistic_cost = []
    seed_costs = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(' : ')[1].strip())
            planner = parts[1].split(' : ')[1].strip()
            cost = float(parts[2].split(' : ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"learned": None, "optimistic": None}
            seed_costs[seed][planner] = cost

    for seed in sorted(seed_costs.keys()):
        learned_cost.append(seed_costs[seed]["learned"])
        optimistic_cost.append(seed_costs[seed]["optimistic"])

    return learned_cost, optimistic_cost, seed_costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/mr_task/mr_task_eval')
    parser.add_argument('--num_robots', type=int, default=2)
    args = parser.parse_args()
    file_path = Path(args.save_dir) / f'log_{args.num_robots}.txt'

    learned_cost, optimistic_cost, seed_costs = extract_costs(file_path)
    assert len(learned_cost) == len(optimistic_cost)
    mrlsp.utils.plotting.make_scatter_plot_with_box(optimistic_cost, learned_cost, xlabel='Optimistic', ylabel='Learned')
    image_name = Path(args.save_dir) / f'log_{args.num_robots}_scatter.png'
    plt.tight_layout()
    plt.savefig(image_name)
