import re
import numpy as np
import matplotlib.pyplot as plt
import lsp
import mrlsp
from pathlib import Path


if __name__ == "__main__":
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--num_robots', type=int, default=1)
    parser.add_argument('--eval_dir', type=str, default='log')
    parser.add_argument('--logfile_name', type=str, default='log')
    args = parser.parse_args()

    baseline_cost = []
    learned_cost = []
    seeds = []

    # Get the data from the log file
    regex_match = re.compile(r"SEED\s*:\s*(\d+)\s*\|\s*learned\s*:\s*([\d.]+)\s*\|\s*optimistic\s*:\s*([\d.]+)")
    logfile = Path(args.save_dir).parent / f'{args.logfile_name}_robots_{args.num_robots}_gcn.txt'
    with open(logfile) as f:
        for line in f:
            match = regex_match.match(line)
            if match:
                seeds.append(int(match.group(1)))
                learned_cost.append(float(match.group(2)))
                baseline_cost.append(float(match.group(3)))
    assert len(seeds) == len(learned_cost) == len(baseline_cost)

    # Plot the scatter plot
    mrlsp.utils.plotting.make_scatter_plot_with_box(baseline_cost, learned_cost, xlabel="Optimistic", ylabel="MR-LSP")
    image_name = Path(args.save_dir) / f'r_{args.num_robots}_scatter.png'
    plt.tight_layout()
    plt.savefig(image_name)
