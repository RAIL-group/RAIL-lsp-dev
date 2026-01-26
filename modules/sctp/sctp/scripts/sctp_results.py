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
    seed_costs = {}
    seed_truntimes = {}
    seed_steptimes = {}
    seed_samptimes = {}
    seed_spolicytimes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(': ')[1].strip())
            planner = parts[2].split(': ')[1].strip()
            cost = float(parts[5].split(': ')[1].strip())
            truntime = float(parts[6].split(': ')[1].strip())
            steptime = float(parts[7].split(': ')[1].strip())
            samptime = float(parts[8].split(': ')[1].strip())
            spolicytime = float(parts[9].split(': ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapavp": None, \
                                    "jsapavp2": None, "jsapdap": None, "dsapavp": None}
                seed_truntimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapavp": None, \
                            "jsapavp2": None,"jsapdap": None, "dsapavp": None}
                seed_steptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapavp": None, \
                            "jsapavp2": None,"jsapdap": None, "dsapavp": None}
                seed_samptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapavp": None, \
                            "jsapavp2": None,"jsapdap": None, "dsapavp": None}
                seed_spolicytimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapavp": None, \
                            "jsapavp2": None,"jsapdap": None, "dsapavp": None}
            seed_costs[seed][planner] = cost
            seed_truntimes[seed][planner] = truntime
            seed_steptimes[seed][planner] = steptime
            seed_samptimes[seed][planner] = samptime 
            seed_spolicytimes[seed][planner] = spolicytime

    return seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seed_spolicytimes
def get_planner_data(seed_costs):
    jsap = []
    jsap2 = []
    base = []
    jsapavp = []
    jsapavp2 = []
    jsapdap = []
    dsapavp = []
    for seed in sorted(seed_costs.keys()):
        jsap.append(seed_costs[seed]["jsap"])
        jsap2.append(seed_costs[seed]["jsap2"])
        base.append(seed_costs[seed]["ctp"])
        jsapavp.append(seed_costs[seed]["jsapavp"])
        jsapavp2.append(seed_costs[seed]["jsapavp2"])
        jsapdap.append(seed_costs[seed]["jsapdap"])
        dsapavp.append(seed_costs[seed]["dsapavp"])
    return base, jsap, jsapavp, jsapdap, dsapavp, jsap2, jsapavp2

def plot_scatter_data(file_path, args):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seeds_policytimes = extract_costs(file_path)
    assert len(seed_costs) == len(seed_truntimes)    
    base_dist, jsap_dist, jsapavp_dist, jsapdap_dist, dsapavp_dist, jsap2_dist, jsapavp2_dist = get_planner_data(seed_costs)
    base_ttimes, jsap_ttimes, jsapavp_ttimes, jsapdap_ttimes, dsapavp_ttimes, jsap2_ttimes, \
                jsapavp2_ttimes = get_planner_data(seed_truntimes)
    base_steptimes, jsap_steptimes, jsapavp_steptimes, jsapdap_steptimes, dsapavp_steptimes, jsap2_steptimes, \
                jsapavp2_steptimes = get_planner_data(seed_steptimes)
    base_samptimes, jsap_samptimes, jsapavp_samptimes, jsapdap_samptimes, dsapavp_samptimes, jsap2_samptimes, \
                jsapavp2_samptimes = get_planner_data(seed_samptimes)
    base_spolicytimes, jsap_spolicytimes, jsapavp_spolicytimes, jsapdap_spolicytimes, dsapavp_spolicytimes, jsap2_spolicytimes, \
                jsapavp2_spolicytimes = get_planner_data(seeds_policytimes)
    
    print(f"CTP: costs: {np.average(base_dist):0.2f}, "\
          f"steptime: {np.average(base_steptimes):0.2f}, samptime: {np.average(base_samptimes):0.2f}, "\
          f"spolicytime: {np.average(base_spolicytimes):0.2f}, total runtime: {np.average(base_ttimes):0.2f}")
    
    if jsap_dist[0] != None:
        # print(f"jsap dist data: {jsap_dist}")
        plotting.make_scatter_plot_with_box(base_dist, jsap_dist, xlabel='CTP', ylabel='JSAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_jsap_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"JSAP: costs: {np.average(jsap_dist):0.2f}, "\
          f"steptime: {np.average(jsap_steptimes):0.2f}, samptime: {np.average(jsap_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsap_spolicytimes):0.2f}, total runtime: {np.average(jsap_ttimes):0.2f}")
        
    
    if jsap2_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsap2_dist, xlabel='CTP', ylabel='JSAP2')
        args.num_drones = 2
        image_name = Path(args.save_dir) / f'plot_cost_ctp_jsap2_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"JSAP2: costs: {np.average(jsap2_dist):0.2f},  "\
          f"steptime: {np.average(jsap2_steptimes):0.2f}, samptime: {np.average(jsap2_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsap2_spolicytimes):0.2f}, total runtime: {np.average(jsap2_ttimes):0.2f}")
    
    if jsapavp_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapavp_dist, xlabel='CTP', ylabel='JSAP-AVP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_jsapavp_{args.num_drones}UAVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"JSAPAVP: costs: {np.average(jsapavp_dist):0.2f},  "\
          f"steptime: {np.average(jsapavp_steptimes):0.2f}, samptime: {np.average(jsapavp_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapavp_spolicytimes):0.2f}, total runtime: {np.average(jsapavp_ttimes):0.2f}")
    
    if jsapavp2_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapavp2_dist, xlabel='CTP', ylabel='JSAP-AVP2')
        args.num_drones = 2
        image_name = Path(args.save_dir) / f'plot_cost_ctp_jsapavp2_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"JSAPAVP2: costs: {np.average(jsapavp2_dist):0.2f},  "\
          f"steptime: {np.average(jsapavp2_steptimes):0.2f}, samptime: {np.average(jsapavp2_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapavp2_spolicytimes):0.2f}, total runtime: {np.average(jsapavp2_ttimes):0.2f}")
    
    if jsapdap_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapdap_dist, xlabel='CTP', ylabel='SAP-DAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_jsapdap_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"JSAP-DAP: costs: {np.average(jsapdap_dist):0.2f},  "\
          f"steptime: {np.average(jsapdap_steptimes):0.2f}, samptime: {np.average(jsapdap_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapdap_spolicytimes):0.2f}, total runtime: {np.average(jsapdap_ttimes):0.2f}")
    
    if dsapavp_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, dsapavp_dist, xlabel='CTP', ylabel='DSAP-AVP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_dsapavp_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"DSAPAVP: costs: {np.average(dsapavp_dist):0.2f},  "\
          f"steptime: {np.average(dsapavp_steptimes):0.2f}, samptime: {np.average(dsapavp_samptimes):0.2f}, "\
          f"spolicytime: {np.average(dsapavp_spolicytimes):0.2f}, total runtime: {np.average(dsapavp_ttimes):0.2f}")
    
    

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
    parser.add_argument('--num_drones', type=int, default=2)
    parser.add_argument('--num_ugvs', type=int, default=1)
    
    args = parser.parse_args()
    scatter_data = True
    file_path = args.save_dir
    args.num_ugvs = 1
    
    if scatter_data:
        file_path = Path(file_path)/ f'results_{args.num_ugvs}UGVs.txt'
        plot_scatter_data(file_path, args)
    else:
        plot_data_varying_drones(file_path)
    


    