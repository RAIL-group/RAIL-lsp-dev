import numpy as np
import argparse
from sctp.utils import plotting
from pathlib import Path
import matplotlib.pyplot as plt


def extract_costs(file_path):
    # print(f"Extr costs from {file_path}")
    sctp_cost = []
    base_cost = []
    sctpiv_cost = []
    sctpfk_cost = []
    sctpivfk_cost = []
    base_runtime = []
    sctp_runtime = []
    sctpiv_runtime = []
    sctpfk_runtime = []
    sctpivfk_runtime = []
    sctpivtwoact_cost = []
    seed_costs = {}
    seed_runtimes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(' : ')[1].strip())
            planner = parts[1].split(' : ')[1].strip()
            cost = float(parts[3].split(' : ')[1].strip())
            runtime = float(parts[4].split(' : ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"sctp": None, "base": None, "sctpiv": None, "sctpfk": None, "sctpivfk": None, "sctpivtwoact": None}
                seed_runtimes[seed] = {"sctp": None, "base": None, "sctpiv": None, "sctpfk": None, "sctpivfk": None, "sctpivtwoact": None}
            seed_costs[seed][planner] = cost
            seed_runtimes[seed][planner] = runtime

    for seed in sorted(seed_costs.keys()):
        sctp_cost.append(seed_costs[seed]["sctp"])
        base_cost.append(seed_costs[seed]["base"])
        sctpiv_cost.append(seed_costs[seed]["sctpiv"])
        # sctpfk_cost.append(seed_costs[seed]["sctpfk"])
        # sctpivfk_cost.append(seed_costs[seed]["sctpivfk"])
        # sctpivtwoact_cost.append(seed_costs[seed]["sctpivtwoactfk"])
        sctp_runtime.append(seed_runtimes[seed]["sctp"])
        base_runtime.append(seed_runtimes[seed]["base"])
        sctpiv_runtime.append(seed_runtimes[seed]["sctpiv"])
        # sctpfk_runtime.append(seed_runtimes[seed]["sctpfk"])
        # sctpivfk_runtime.append(seed_runtimes[seed]["sctpivfk"])

    return base_cost, sctp_cost, sctpfk_cost, sctpiv_cost, sctpivfk_cost, sctpivtwoact_cost, \
           base_runtime, sctp_runtime, sctpfk_runtime, sctpiv_runtime, sctpivfk_runtime,  \
            seed_costs, seed_runtimes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp/sctp_eval')
    parser.add_argument('--num_drones', type=int, default=1)
    
    args = parser.parse_args()
    # print(args.save_dir)
    file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'


    base_cost, sctp_cost, sctpfk_cost, sctpiv_cost, sctpivfk_cost, sctpivtwoact_cost, \
           base_runtime, sctp_runtime, sctpfk_runtime, sctpiv_runtime, sctpivfk_runtime,  \
            seed_costs, seed_runtimes = extract_costs(file_path)
    # sctp_cost, base_cost, sctpiv_cost, sctp_runtime, base_runtime, \
    #         sctpiv_runtime, seed_costs, seed_runtimes = extract_costs(file_path)
    assert len(sctp_cost) == len(base_cost)
    
    print(f"The number of data {len(sctp_cost)}")

    plotting.make_scatter_plot_with_box(base_cost, sctp_cost, xlabel='Baseline', ylabel='SCTP')
    image_name = Path(args.save_dir) / f'plot_cost_baseline_stcp_{args.num_drones}.png'
    plt.tight_layout()
    plt.savefig(image_name)
    plotting.make_scatter_plot_with_box(base_cost, sctpiv_cost, xlabel='Baseline', ylabel='SCTPIV')
    image_name = Path(args.save_dir) / f'plot_cost_baseline_sctpiv_{args.num_drones}.png'
    plt.tight_layout()
    plt.savefig(image_name)
    # plotting.make_scatter_plot_with_box(base_cost, sctpivtwoact_cost, xlabel='Baseline', ylabel='SCTPTWOACTFK')
    # image_name = Path(args.save_dir) / f'plot_cost_baseline_stcpivtwoactfk_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    # plotting.make_scatter_plot_with_box(sctp_cost, sctpivtwoact_cost, xlabel='SCTP', ylabel='SCTPTWOACT')
    # image_name = Path(args.save_dir) / f'plot_cost_sctp_sctpivtwoact_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    
    # plotting.make_scatter_plot_with_box(sctp_cost, sctpiv_cost, xlabel='SCTP', ylabel='SCTPIV')
    # image_name = Path(args.save_dir) / f'plot_cost_sctp_sctpiv_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    
    # plotting.make_scatter_plot_with_box(base_runtime, sctp_runtime, xlabel='Baseline', ylabel='SCTP')
    # image_name = Path(args.save_dir) / f'plot_runtime_baseline_stcp_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    # plotting.make_scatter_plot_with_box(base_runtime, sctpiv_runtime, xlabel='Baseline', ylabel='SCTPIV')
    # image_name = Path(args.save_dir) / f'plot_runtime_baseline_sctpiv_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    # plotting.make_scatter_plot_with_box(sctp_runtime, sctpiv_runtime, xlabel='SCTP', ylabel='SCTPIV')
    # image_name = Path(args.save_dir) / f'plot_runtime_sctp_sctpiv_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)