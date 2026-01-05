import numpy as np
import argparse
from sctp.utils import plotting
from pathlib import Path
import matplotlib.pyplot as plt


def sctp_costs_runtimes(file_path, ugv_num=1, drone_nums = [0,1,2,3]):    
    seed_costs = {}
    seed_aruntimes = {}
    seed_plantimes = {}
    for drone_num in drone_nums:
        file_name = Path(file_path) / f'log_{ugv_num}_{drone_num}.txt'
        print(f"Extract costs and runtimes from {file_name}")
        with open(file_name, 'r') as file:
            for line in file:
                parts = line.split(' | ')
                seed = int(parts[0].split(' : ')[1].strip())
                planner = parts[1].split(' : ')[1].strip()
                cost = float(parts[3].split(' : ')[1].strip())
                plantime = float(parts[4].split(' : ')[1].strip())
                aruntime = float(parts[6].split(' : ')[1].strip())
                
                if seed not in seed_costs:
                    seed_costs[seed] = {"base": None, "jsctp1": None, "jsctp2": None, "dsctp1": None, "dsctp2": None, "dsctp3": None,
                                        "base_aa": None, "jsctp1_aa": None, "jsctp2_aa": None, "dsctp1_aa": None, "dsctp2_aa": None, "dsctp3_aa": None}
                    seed_aruntimes[seed] = {"base": None, "jsctp1": None, "jsctp2": None, "dsctp1": None, "dsctp2": None, "dsctp3": None,
                                           "base_aa": None, "jsctp1_aa": None, "jsctp2_aa": None, "dsctp1_aa": None, "dsctp2_aa": None, "dsctp3_aa": None}
                    seed_plantimes[seed] = {"base": None, "jsctp1": None, "jsctp2": None, "dsctp1": None, "dsctp2": None, "dsctp3": None,
                                           "base_aa": None, "jsctp1_aa": None, "jsctp2_aa": None, "dsctp1_aa": None, "dsctp2_aa": None, "dsctp3_aa": None}
                seed_costs[seed][planner] = cost
                seed_aruntimes[seed][planner] = aruntime
                seed_plantimes[seed][planner] = plantime
    return seed_costs, seed_aruntimes, seed_plantimes


def extract_costs(file_path):
    # print(f"Extr costs from {file_path}")
    jsap_cost = []
    base_cost = []
    jsapavp_cost = []
    dsap_cost = []
    dsapavp_cost = []
    base_runtime = []
    jsap_runtime = []
    jsapavp_runtime = []
    dsap_runtime = []
    dsapavp_runtime = []
    seed_costs = {}
    seed_runtimes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(': ')[1].strip())
            planner = parts[2].split(': ')[1].strip()
            cost = float(parts[4].split(': ')[1].strip())
            runtime = float(parts[6].split(': ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"ctp": None, "jsap": None, "jsapavp": None, "dsap": None, "dsapavp": None}
                seed_runtimes[seed] = {"ctp": None, "jsap": None, "jsapavp": None, "dsap": None, "dsapavp": None}
            seed_costs[seed][planner] = cost
            seed_runtimes[seed][planner] = runtime

    for seed in sorted(seed_costs.keys()):
        jsap_cost.append(seed_costs[seed]["jsap"])
        base_cost.append(seed_costs[seed]["ctp"])
        jsapavp_cost.append(seed_costs[seed]["jsapavp"])
        dsap_cost.append(seed_costs[seed]["dsap"])
        dsapavp_cost.append(seed_costs[seed]["dsapavp"])
        jsap_runtime.append(seed_runtimes[seed]["jsap"])
        base_runtime.append(seed_runtimes[seed]["ctp"])
        jsapavp_runtime.append(seed_runtimes[seed]["jsapavp"])
        dsap_runtime.append(seed_runtimes[seed]["dsap"])
        dsapavp_runtime.append(seed_runtimes[seed]["dsapavp"])

    return base_cost, jsap_cost, jsapavp_cost, dsap_cost, dsapavp_cost, \
           base_runtime, jsap_runtime, jsapavp_runtime, dsap_runtime, dsapavp_runtime,  \
            seed_costs, seed_runtimes

def plot_scatter_data(file_path):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    base, jsap, jsapavp, dsap, dsapavp, \
    base_runtime, jsap_runtime, jsapavp_runtime, dsap_runtime, dsapavp_runtime,  \
            seed_costs, seed_runtimes = extract_costs(file_path)
    assert len(jsap) == len(base)    
    # print(f"The number of data {len(jsap)} {len(dsap)}")
    # plotting.make_scatter_plot_with_box(base_cost, sctp_cost, xlabel='Baseline', ylabel='SCTP')
    # image_name = Path(args.save_dir) / f'plot_cost_baseline_jstcp1.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    plotting.make_scatter_plot_with_box(base, jsap, xlabel='CTP', ylabel='JSAP')
    image_name = Path(args.save_dir) / f'plot_cost_base_jsap_{args.num_ugvs}UGVs.png'
    plt.tight_layout()
    plt.savefig(image_name)
    
    plotting.make_scatter_plot_with_box(base, jsapavp, xlabel='CTP', ylabel='JSAP-AVP')
    image_name = Path(args.save_dir) / f'plot_cost_base_jsapavp_{args.num_drones}UAVs.png'
    plt.tight_layout()
    plt.savefig(image_name)
    
    # plotting.make_scatter_plot_with_box(base, dsap, xlabel='Baseline', ylabel='DSAP')
    # image_name = Path(args.save_dir) / f'plot_cost_base_dsap{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    
    # plotting.make_scatter_plot_with_box(base, dsapavp, xlabel='Baseline', ylabel='DSAP-AVP')
    # image_name = Path(args.save_dir) / f'plot_cost_base_dsapavp{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    
    

def plot_data_varying_drones(file_path, ugv_nums=[1,2], drone_nums = [0,1,2], plot_costs=True,
                             plot_runtimes=False, plot_plantimes=False):
    for ugv_num in ugv_nums:
        costs, rtimes, ptimes = sctp_costs_runtimes(file_path, ugv_num=ugv_nums, drone_nums=drone_nums)
        base_data = []
        jsctp1_data = []
        jsctp2_data = []
        dsctp1_data = []
        dsctp2_data = []
        dsctp3_data = []
        base_aa_data = []
        jsctp1_aa_data = []
        jsctp2_aa_data = []
        dsctp1_aa_data = []
        dsctp2_aa_data = []
        dsctp3_aa_data = []
    if plot_costs:
        for seed in sorted(costs.keys()):
            base_data.append(costs[seed]["base"])
            jsctp1_data.append(costs[seed]["jsctp1"])
            jsctp2_data.append(costs[seed]["jsctp2"])
            dsctp1_data.append(costs[seed]["dsctp1"])
            dsctp2_data.append(costs[seed]["dsctp2"])
            dsctp3_data.append(costs[seed]["dsctp3"])
            base_aa_data.append(costs[seed]["base_aa"])
            jsctp1_aa_data.append(costs[seed]["jsctp1_aa"])
            jsctp2_aa_data.append(costs[seed]["jsctp2_aa"])
            dsctp1_aa_data.append(costs[seed]["dsctp1_aa"])
            dsctp2_aa_data.append(costs[seed]["dsctp2_aa"])
            dsctp3_aa_data.append(costs[seed]["dsctp3_aa"])
    elif plot_runtimes:
        assert plot_costs is False, "Plot either costs or runtimes"
        for seed in sorted(rtimes.keys()):
            base_data.append(rtimes[seed]["base"])
            jsctp1_data.append(rtimes[seed]["jsctp1"])
            jsctp2_data.append(rtimes[seed]["jsctp2"])
            dsctp1_data.append(rtimes[seed]["dsctp1"])
            dsctp2_data.append(rtimes[seed]["dsctp2"])
            dsctp3_data.append(rtimes[seed]["dsctp3"])
            base_aa_data.append(rtimes[seed]["base_aa"])
            jsctp1_aa_data.append(rtimes[seed]["jsctp1_aa"])
            jsctp2_aa_data.append(rtimes[seed]["jsctp2_aa"])
            dsctp1_aa_data.append(rtimes[seed]["dsctp1_aa"])
            dsctp2_aa_data.append(rtimes[seed]["dsctp2_aa"])
            dsctp3_aa_data.append(rtimes[seed]["dsctp3_aa"])
    elif plot_plantimes:
        assert plot_costs is False, "Plot either costs or runtimes"
        for seed in sorted(rtimes.keys()):
            base_data.append(ptimes[seed]["base"])
            jsctp1_data.append(ptimes[seed]["jsctp1"])
            jsctp2_data.append(ptimes[seed]["jsctp2"])
            dsctp1_data.append(ptimes[seed]["dsctp1"])
            dsctp2_data.append(ptimes[seed]["dsctp2"])
            dsctp3_data.append(ptimes[seed]["dsctp3"])
            base_aa_data.append(ptimes[seed]["base_aa"])
            jsctp1_aa_data.append(ptimes[seed]["jsctp1_aa"])
            jsctp2_aa_data.append(ptimes[seed]["jsctp2_aa"])
            dsctp1_aa_data.append(ptimes[seed]["dsctp1_aa"])
            dsctp2_aa_data.append(ptimes[seed]["dsctp2_aa"])
            dsctp3_aa_data.append(ptimes[seed]["dsctp3_aa"])
        
    
    # base_cost, sctp_cost, sctpfk_cost, sctpiv_cost, sctpivfk_cost, sctpivtwoact_cost, \
    #        base_runtime, sctp_runtime, sctpfk_runtime, sctpiv_runtime, sctpivfk_runtime,  \
    #         seed_costs, seed_runtimes = extract_costs(file_path)
    # sctp_cost, base_cost, sctpiv_cost, sctp_runtime, base_runtime, \
    #         sctpiv_runtime, seed_costs, seed_runtimes = extract_costs(file_path)
    # assert len(sctp_cost) == len(base_cost)
    
    # print(f"The number of data {len(sctp_cost)}")

    # plotting.make_scatter_plot_with_box(base_cost, sctp_cost, xlabel='Baseline', ylabel='SCTP')
    # image_name = Path(args.save_dir) / f'plot_cost_baseline_stcp_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    # plotting.make_scatter_plot_with_box(base_cost, sctpiv_cost, xlabel='Baseline', ylabel='SCTPIV')
    # image_name = Path(args.save_dir) / f'plot_cost_baseline_sctpiv_{args.num_drones}.png'
    # plt.tight_layout()
    # plt.savefig(image_name)
    
    # for drone_num in range(0, 4):
    #     file_name = Path(args.save_dir) / f'log_{args.num_drones}_{drone_num}.txt'
    #     print(f"Extract costs from {file_name}")
    #     base_cost, sctp_cost,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp/sctp_eval')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_ugvs', type=int, default=1)
    
    args = parser.parse_args()
    scatter_data = True
    file_path = args.save_dir
    
    if scatter_data:
        file_path = Path(file_path)/ f'results_{args.num_ugvs}UGVs.txt'
        plot_scatter_data(file_path)
    else:
        plot_data_varying_drones(file_path)
    


    