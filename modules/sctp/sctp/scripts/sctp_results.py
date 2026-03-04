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
                seed_costs[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                                    "jsapiap2": None, "jsapdap": None, "jsapdap2": None}
                seed_truntimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None}
                seed_steptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None}
                seed_samptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None}
                seed_spolicytimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None}
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
        jsapavp.append(seed_costs[seed]["jsapiap"])
        jsapavp2.append(seed_costs[seed]["jsapiap2"])
        jsapdap.append(seed_costs[seed]["jsapdap"])
        dsapavp.append(seed_costs[seed]["jsapdap2"])
    return base, jsap, jsapavp, jsapdap, dsapavp, jsap2, jsapavp2

def plot_scatter_data(file_path, args):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seeds_policytimes = extract_costs(file_path)
    assert len(seed_costs) == len(seed_truntimes)
    base_dist, jsap_dist, jsapavp_dist, jsapdap_dist, jsapdap2_dist, jsap2_dist, jsapavp2_dist = get_planner_data(seed_costs)
    base_ttimes, jsap_ttimes, jsapavp_ttimes, jsapdap_ttimes, jsapdap2_ttimes, jsap2_ttimes, \
                jsapavp2_ttimes = get_planner_data(seed_truntimes)
    base_steptimes, jsap_steptimes, jsapavp_steptimes, jsapdap_steptimes, jsapdap2_steptimes, jsap2_steptimes, \
                jsapavp2_steptimes = get_planner_data(seed_steptimes)
    base_samptimes, jsap_samptimes, jsapavp_samptimes, jsapdap_samptimes, jsapdap2_samptimes, jsap2_samptimes, \
                jsapavp2_samptimes = get_planner_data(seed_samptimes)
    base_spolicytimes, jsap_spolicytimes, jsapavp_spolicytimes, jsapdap_spolicytimes, jsapdap2_spolicytimes, jsap2_spolicytimes, \
                jsapavp2_spolicytimes = get_planner_data(seeds_policytimes)
    
    print(base_dist)
    print(jsap_dist)
    if base_dist[0] != None:
        print(f"CTP: costs: {np.average(base_dist):0.2f}, "\
          f"steptime: {np.average(base_steptimes):0.2f}, samptime: {np.average(base_samptimes):0.2f}, "\
          f"spolicytime: {np.average(base_spolicytimes):0.2f}, total runtime: {np.average(base_ttimes):0.2f}")
    
    if jsap_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsap_dist, xlabel='CTP', ylabel='SAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sap_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAP: costs: {np.average(jsap_dist):0.2f}, "\
          f"steptime: {np.average(jsap_steptimes):0.2f}, samptime: {np.average(jsap_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsap_spolicytimes):0.2f}, total runtime: {np.average(jsap_ttimes):0.2f}")
        
    
    if jsap2_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsap2_dist, xlabel='CTP', ylabel='SAP2')
        args.num_drones = 2
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sap2_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAP2: costs: {np.average(jsap2_dist):0.2f},  "\
          f"steptime: {np.average(jsap2_steptimes):0.2f}, samptime: {np.average(jsap2_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsap2_spolicytimes):0.2f}, total runtime: {np.average(jsap2_ttimes):0.2f}")
    
    if jsapavp_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapavp_dist, xlabel='CTP', ylabel='SAP-IAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sapiap_{args.num_drones}UAVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAPIAP: costs: {np.average(jsapavp_dist):0.2f},  "\
          f"steptime: {np.average(jsapavp_steptimes):0.2f}, samptime: {np.average(jsapavp_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapavp_spolicytimes):0.2f}, total runtime: {np.average(jsapavp_ttimes):0.2f}")
    
    if jsapavp2_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapavp2_dist, xlabel='CTP', ylabel='SAP-IAP2')
        args.num_drones = 2
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sapiap2_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAPIAP2: costs: {np.average(jsapavp2_dist):0.2f},  "\
          f"steptime: {np.average(jsapavp2_steptimes):0.2f}, samptime: {np.average(jsapavp2_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapavp2_spolicytimes):0.2f}, total runtime: {np.average(jsapavp2_ttimes):0.2f}")
    
    if jsapdap_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapdap_dist, xlabel='CTP', ylabel='SAP-DAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sapdap_{args.num_ugvs}UGVs.pdf'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAP-DAP: costs: {np.average(jsapdap_dist):0.2f},  "\
          f"steptime: {np.average(jsapdap_steptimes):0.2f}, samptime: {np.average(jsapdap_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapdap_spolicytimes):0.2f}, total runtime: {np.average(jsapdap_ttimes):0.2f}")
    
        plotting.make_scatter_plot_with_box(jsapdap_dist, jsapavp_dist, xlabel='SAP-DAP', ylabel='SAP-IAP')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_sapdap_sapiap_{args.num_ugvs}UGVs.pdf='
        plt.tight_layout()
        plt.savefig(image_name)
    
    if jsapdap2_dist[0] != None:
        plotting.make_scatter_plot_with_box(base_dist, jsapdap2_dist, xlabel='CTP', ylabel='SAP-DAP2')
        args.num_drones = 1
        image_name = Path(args.save_dir) / f'plot_cost_ctp_sapdap2_{args.num_ugvs}UGVs.png'
        plt.tight_layout()
        plt.savefig(image_name)
        print(f"SAPDAP2: costs: {np.average(jsapdap2_dist):0.2f},  "\
          f"steptime: {np.average(jsapdap2_steptimes):0.2f}, samptime: {np.average(jsapdap2_samptimes):0.2f}, "\
          f"spolicytime: {np.average(jsapdap2_spolicytimes):0.2f}, total runtime: {np.average(jsapdap2_ttimes):0.2f}")
    

def processed_data(input_file, output_file, args, prefix=None):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seeds_policytimes = extract_costs(input_file)
    assert len(seed_costs) == len(seed_truntimes)
    base_dist, jsap_dist, jsapavp_dist, jsapdap_dist, jsapdap2_dist, jsap2_dist, jsapavp2_dist = get_planner_data(seed_costs)
    base_ttimes, jsap_ttimes, jsapavp_ttimes, jsapdap_ttimes, jsapdap2_ttimes, jsap2_ttimes, \
                jsapavp2_ttimes = get_planner_data(seed_truntimes)
    base_steptimes, jsap_steptimes, jsapavp_steptimes, jsapdap_steptimes, jsapdap2_steptimes, jsap2_steptimes, \
                jsapavp2_steptimes = get_planner_data(seed_steptimes)
    base_samptimes, jsap_samptimes, jsapavp_samptimes, jsapdap_samptimes, jsapdap2_samptimes, jsap2_samptimes, \
                jsapavp2_samptimes = get_planner_data(seed_samptimes)
    
    # print(base_dist)
    # print(jsap_dist)
    if base_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: CTP        | UGVs: {args.num_ugvs} | AVG_COST: {np.average(base_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(base_steptimes):0.2f} | SAMP_TIME: {np.average(base_samptimes):0.2f} |"
                f" T.TIME: {np.average(base_ttimes):0.2f} \n")    
   
    if jsap_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: JSAP-1     | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsap_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(jsap_steptimes):0.2f} | SAMP_TIME: {np.average(jsap_samptimes):0.2f} |"
                f" T.TIME: {np.average(jsap_ttimes):0.2f} \n")    
    
    if jsap2_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: JSAP-2     | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsap2_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(jsap2_steptimes):0.2f} | SAMP_TIME: {np.average(jsap2_samptimes):0.2f} |"
                f" T.TIME: {np.average(jsap2_ttimes):0.2f} \n")    
    
        
    if jsapavp_dist[0] != None:
        with open(output_file, "a+") as f:
            data = f"PLANNER: JSAP-IAP-1 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapavp_dist):0.2f} |"\
                f" AVG_STEP_TIME: {np.average(jsapavp_steptimes):0.2f} | SAMP_TIME: {np.average(jsapavp_samptimes):0.2f} |"\
                f" T.TIME: {np.average(jsapavp_ttimes):0.2f}"
            if prefix is not None:
                data = data + prefix
            else:
                data = data + "\n"
            f.write(data)
            # f.write(f"PLANNER: JSAP-IAP-1 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapavp_dist):0.2f} |"
            #     f" AVG_STEP_TIME: {np.average(jsapavp_steptimes):0.2f} | SAMP_TIME: {np.average(jsapavp_samptimes):0.2f} |"
            #     f" T.TIME: {np.average(jsapavp_ttimes):0.2f} \n")
    
    if jsapavp2_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: JSAP-IAP-2 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapavp2_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(jsapavp2_steptimes):0.2f} | SAMP_TIME: {np.average(jsapavp2_samptimes):0.2f} |"
                f" T.TIME: {np.average(jsapavp2_ttimes):0.2f} \n")

    if jsapdap_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: JSAP-DAP-1 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapdap_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(jsapdap_steptimes):0.2f} | SAMP_TIME: {np.average(jsapdap_samptimes):0.2f} |"
                f" T.TIME: {np.average(jsapdap_ttimes):0.2f} \n")
    
    if jsapdap2_dist[0] != None:
        with open(output_file, "a+") as f:
            f.write(f"PLANNER: JSAP-DAP-2 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapdap2_dist):0.2f} |"
                f" AVG_STEP_TIME: {np.average(jsapdap2_steptimes):0.2f} | SAMP_TIME: {np.average(jsapdap2_samptimes):0.2f} |"
                f" T.TIME: {np.average(jsapdap2_ttimes):0.2f} \n")
    
# def plot_madist_allinOne_std(x, data, std, featureNames, yName, 
#                     ranges, rangeStep, envName, xName=None, outpath=None):
#     markers = ['o', '^',  'D', 's', 'p', '*', 'h']
#     colors = ['red', 'magenta', 'orange', 'blue', 'green', 'purple', 'brown']
#     order = 10
#     for ii, dat in enumerate(data):         
#         plt.errorbar(x, dat, std,
#                     color=colors[ii],   
#                     linewidth=2.0, 
#                     # linestyle='None', 
#                     marker=markers[ii],
#                     markersize=9,
#                     label=featureNames[ii],
#                     zorder=order
#                 )
                    
#     plt.title(envName, fontsize=18)
#     plt.ylim(ranges[0], ranges[1])
#     major_ticks = np.arange(ranges[0], ranges[1], rangeStep)
#     plt.grid(axis = "y")
#     plt.legend(fontsize=12, loc='best')
    
#     # for runtime and travel distances
#     plt.xticks(x,visible=True,fontsize=12)
#     if xName is not None:
#         plt.xlabel(xName, fontsize=15)
#     else:
#         plt.xlabel('number of UGVs', fontsize=15)
#     plt.ylabel(yName, fontsize=15)
#     plt.yticks(major_ticks,fontsize=12)
#     if outpath is not None:
#         plt.savefig(outpath, bbox_inches='tight')
#     else:
#         plt.savefig(f'/data/sctp/Jan30/plot_{envName}_{xName}.pdf', bbox_inches='tight')

def read_processed_data(filepath, prune_actions=False):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split('|')
            planner = parts[0].split(':')[1].strip()
            ugvs = int(parts[1].split(':')[1].strip())
            avg_cost = float(parts[2].split(':')[1].strip())
            avg_step_time = float(parts[3].split(':')[1].strip())
            samp_time = float(parts[4].split(':')[1].strip())
            total_time = float(parts[5].split(':')[1].strip())
            if len(parts) > 6:
                prune_num_action = int(parts[6].split(':')[1].strip())
                # print(f"The number of prune_action: {prune_num_action}")
            if planner not in data:
                data[planner] = {}
            
            if prune_actions:
                assert len(parts) > 6, "Prune num action is specified but not found in the data"
                data[planner][prune_num_action] = {
                    'avg_cost': avg_cost,
                    'avg_step_time': avg_step_time,
                    'samp_time': samp_time,
                    'total_time': total_time,
                }
            else:
                data[planner][ugvs] = {
                    'avg_cost': avg_cost,
                    'avg_step_time': avg_step_time,
                    'samp_time': samp_time,
                    'total_time': total_time,
                }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp/sctp_eval')
    parser.add_argument('--num_drones', type=int, default=2)
    parser.add_argument('--num_ugvs', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='prune_num_action')
    
    args = parser.parse_args()
    prune_num_actions = None
    scatter_data = False
    need_to_process = False
    plot_all = False
    if args.exp_name == 'prune_num_action':
        prune_num_actions = [1,2,3,4,5]
    elif args.exp_name == 'statistics':
        scatter_data = True
    elif args.exp_name == 'plot_all':
        plot_all = True
    else:
        raise ValueError(f"Unknown exp_name {args.exp_name}")
    
    file_path = args.save_dir
    
    if scatter_data:
        args.num_ugvs = 1
        input_path = Path(file_path)/ f'plot_data/results_{args.num_ugvs}UGVs.txt'
        plot_scatter_data(input_path, args)
    elif plot_all:
        graphs = ['random', 'bridges', 'islands']
        ugvs_num = [1,2,3]
        if need_to_process:
            for graph in graphs:
                output_path = Path(file_path)/ f'{graph}/processed_results.txt'    
                for ugv_num in ugvs_num:
                    args.num_ugvs = ugv_num
                    input_path = Path(file_path)/ f'{graph}/{ugv_num}/results_{ugv_num}UGVs.txt'
                    processed_data(input_path, output_path, args)
        for graph in graphs:
            output_path = Path(file_path)/ f'{graph}/processed_results.txt'    
            data = read_processed_data(output_path)
            distances = [[data[planner][ugv_num]['avg_cost'] for ugv_num in ugvs_num] for planner in data]
            plotting.plot_madist_allinOne_std(x=ugvs_num, data=distances, std=None, featureNames=list(data.keys()), yName="Distances [m]", 
                    ranges=(150, 1400), rangeStep=200, envName=graph)
            
            # distances = [[data[planner][ugv_num]['avg_cost']/float(ugv_num) for ugv_num in ugvs_num] for planner in data]
            # plot_madist_allinOne_std(x=ugvs_num, data=distances, std=None, featureNames=list(data.keys()), yName="Distances [m]", 
            #         ranges=(150, 700), rangeStep=100, envName=graph, axis=plt.subplots(1, 1)[1], i=graphs.index(graph))
            # plt.show()
    elif prune_num_actions is not None:
        output_filename = Path(file_path) / f'bridges/max_uav_action/processed_data.txt'
        planner  = 'JSAP-IAP-1'
        args.num_ugvs = 1
        is_data_processed = True
        if not is_data_processed:
            for prune_num_action in prune_num_actions:
                input_filename = Path(file_path) / f'bridges/max_uav_action/1_{prune_num_action}/results_1UGVs.txt'
                prefix = f" | PRUNE_NUM_ACTION: {prune_num_action}\n"
                processed_data(input_file=input_filename, output_file=output_filename, args=args, prefix=prefix)
        
        data = read_processed_data(output_filename, prune_actions=True)
        distances = [data[planner][prune_num_action]['avg_cost'] for prune_num_action in prune_num_actions]
        figure_out = Path(file_path) / f'bridges/max_uav_action/prune_actions_fig.pdf'
        plotting.plot_madist_allinOne_std(x=prune_num_actions, data=[distances], std=None, featureNames=list(data.keys()), yName="Distances [m]",
                ranges=(200, 400), rangeStep=50, envName=f"Bridges_Graph", xName="Top-k of drone actions", outpath=figure_out)
        plt.show()
    