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
                                    "jsapiap2": None, "jsapdap": None, "jsapdap2": None, "jsapliap": None, "jsapliap2": None}
                seed_truntimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None, "jsapliap": None, "jsapliap2": None}
                seed_steptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None,"jsapliap": None, "jsapliap2": None}
                seed_samptimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None, "jsapliap": None, "jsapliap2": None}
                seed_spolicytimes[seed] = {"ctp": None, "jsap": None, "jsap2": None, "jsapiap": None, \
                            "jsapiap2": None,"jsapdap": None, "jsapdap2": None, "jsapliap": None, "jsapliap2": None}
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
    jsapdap2 = []
    jsapliap = []
    jsapliap2 = []
    
    for seed in sorted(seed_costs.keys()):
        jsap.append(seed_costs[seed]["jsap"])
        jsap2.append(seed_costs[seed]["jsap2"])
        base.append(seed_costs[seed]["ctp"])
        jsapavp.append(seed_costs[seed]["jsapiap"])
        jsapavp2.append(seed_costs[seed]["jsapiap2"])
        jsapdap.append(seed_costs[seed]["jsapdap"])
        jsapdap2.append(seed_costs[seed]["jsapdap2"])
        jsapliap.append(seed_costs[seed]["jsapliap"])
        jsapliap2.append(seed_costs[seed]["jsapliap2"])
    return base, jsap, jsapavp, jsapdap, jsapdap2, jsap2, jsapavp2, jsapliap, jsapliap2

def plot_scatter_data(file_path, args):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    result_file = Path(args.save_dir) / f'costs.txt'
    seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seeds_policytimes = extract_costs(file_path)
    assert len(seed_costs) == len(seed_truntimes)
    base_dist, jsap_dist, jsapavp_dist, jsapdap_dist, jsapdap2_dist, jsap2_dist, jsapavp2_dist,\
               jsapliap_dist, jsapliap2_dist = get_planner_data(seed_costs)
    base_ttimes, jsap_ttimes, jsapavp_ttimes, jsapdap_ttimes, jsapdap2_ttimes, jsap2_ttimes, \
                jsapavp2_ttimes, jsapliap_ttimes, jsapliap2_ttimes = get_planner_data(seed_truntimes)
    base_steptimes, jsap_steptimes, jsapavp_steptimes, jsapdap_steptimes, jsapdap2_steptimes, jsap2_steptimes, \
                jsapavp2_steptimes, jsapliap_steptimes, jsapliap2_steptimes = get_planner_data(seed_steptimes)
    base_samptimes, jsap_samptimes, jsapavp_samptimes, jsapdap_samptimes, jsapdap2_samptimes, jsap2_samptimes, \
                jsapavp2_samptimes, jsapliap_samptimes, jsapliap2_samptimes = get_planner_data(seed_samptimes)
    
    
    planners = ['CTP', 'JSAP', 'JSAP-DAP', 'JSAP-IAP', 'JSAP-LIAP', 'JSAP-2', 'JSAP-DAP2', 'JSAP-IAP2', 'JSAP-LIAP2']
    distances = [base_dist, jsap_dist, jsapdap_dist, jsapavp_dist, jsapliap_dist, jsap2_dist, jsapdap2_dist, \
                    jsapavp2_dist, jsapliap2_dist]
    steptimes = [base_steptimes, jsap_steptimes, jsapdap_steptimes, jsapavp_steptimes, jsapliap_steptimes, jsap2_steptimes,\
                     jsapdap2_steptimes, jsapavp2_steptimes, jsapliap2_steptimes]
    samptimes = [base_samptimes, jsap_samptimes, jsapdap_samptimes, jsapavp_samptimes, jsapliap_samptimes,\
                     jsap2_samptimes, jsapdap2_samptimes, jsapavp2_samptimes, jsapliap2_samptimes]
    ttimes = [base_ttimes, jsap_ttimes, jsapdap_ttimes, jsapavp_ttimes, jsapliap_ttimes,\
                    jsap2_ttimes, jsapdap2_ttimes, jsapavp2_ttimes, jsapliap2_ttimes]
    
    assert distances[0][0] != None
    num_steps = np.average(ttimes[0]) / np.average(steptimes[0]) if np.average(steptimes[0]) > 0.0 else 0.0
    with open(result_file, "a+") as f:
        f.write(f"{planners[0]}: costs: {np.average(distances[0]):0.2f}, "\
                f"samptime_step: {np.average(samptimes[0])/num_steps:0.2f}, steptime: {np.average(steptimes[0]):0.2f}, "\
                f"num_steps: {num_steps:0.1f}, total runtime: {np.average(ttimes[0]):0.2f}\n")
    for i in range(1, len(planners)):
        if distances[i][0] != None:
            steptime_avg = np.average(steptimes[i])
            total_time_avg = np.average(ttimes[i])
            samptime_avg = np.average(samptimes[i])
            num_steps = total_time_avg / steptime_avg if steptime_avg > 0.0 else 0.0
            samptime_step = samptime_avg / num_steps if num_steps > 0.0 else 0.0
            with open(result_file, "a+") as f:
                f.write(f"{planners[i]}: costs: {np.average(distances[i]):0.2f}, "\
                        f"samptime_step: {samptime_step:0.2f}, steptime: {np.average(steptimes[i]):0.2f}, "\
                        f"num_steps: {num_steps:0.1f}, total runtime: {np.average(ttimes[i]):0.2f}\n")
            plotting.make_scatter_plot_with_box(base_dist, distances[i], xlabel='CTP', ylabel=planners[i])
            if (i==2 or i==4 or i==6 or i==8):
                args.num_drones = 2
            else:
                args.num_drones = 1
            image_name = Path(args.save_dir) / f'plot_cost_ctp_{planners[i].lower()}_{args.num_ugvs}UGVs.png'
            plt.tight_layout()
            plt.savefig(image_name)
     

def processed_data(input_file, output_file, args, prefix=None):
    # file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    # print(f"The input file is {input_file}")
    seed_costs, seed_truntimes, seed_steptimes, seed_samptimes, seeds_policytimes = extract_costs(input_file)
    base_dist, jsap_dist, jsapavp_dist, jsapdap_dist, jsapdap2_dist, jsap2_dist, jsapavp2_dist,\
               jsapliap_dist, jsapliap2_dist = get_planner_data(seed_costs)
    base_ttimes, jsap_ttimes, jsapavp_ttimes, jsapdap_ttimes, jsapdap2_ttimes, jsap2_ttimes, \
                jsapavp2_ttimes, jsapliap_ttimes, jsapliap2_ttimes = get_planner_data(seed_truntimes)
    base_steptimes, jsap_steptimes, jsapavp_steptimes, jsapdap_steptimes, jsapdap2_steptimes, jsap2_steptimes, \
                jsapavp2_steptimes, jsapliap_steptimes, jsapliap2_steptimes = get_planner_data(seed_steptimes)
    base_samptimes, jsap_samptimes, jsapavp_samptimes, jsapdap_samptimes, jsapdap2_samptimes, jsap2_samptimes, \
                jsapavp2_samptimes, jsapliap_samptimes, jsapliap2_samptimes = get_planner_data(seed_samptimes)
    
    assert len(seed_costs) == len(seed_truntimes)    
    planners = ['CTP', 'JSAP', 'JSAP-DAP', 'JSAP-IAP', 'JSAP-LIAP', 'JSAP-2', 'JSAP-DAP2', 'JSAP-IAP2', 'JSAP-LIAP2']
    distances = [base_dist, jsap_dist, jsapdap_dist, jsapavp_dist, jsapliap_dist, jsap2_dist, jsapdap2_dist, \
                    jsapavp2_dist, jsapliap2_dist]
    steptimes = [base_steptimes, jsap_steptimes, jsapdap_steptimes, jsapavp_steptimes, jsapliap_steptimes, jsap2_steptimes,\
                     jsapdap2_steptimes, jsapavp2_steptimes, jsapliap2_steptimes]
    samptimes = [base_samptimes, jsap_samptimes, jsapdap_samptimes, jsapavp_samptimes, jsapliap_samptimes,\
                     jsap2_samptimes, jsapdap2_samptimes, jsapavp2_samptimes, jsapliap2_samptimes]
    ttimes = [base_ttimes, jsap_ttimes, jsapdap_ttimes, jsapavp_ttimes, jsapliap_ttimes,\
                    jsap2_ttimes, jsapdap2_ttimes, jsapavp2_ttimes, jsapliap2_ttimes]
    
    for i in range(0, len(planners)):
        if distances[i][0] != None:
            # print(f"The planner is: {planners[i]}")
            steptime_avg = np.average(steptimes[i])
            total_time_avg = np.average(ttimes[i])
            samptime_avg = np.average(samptimes[i])
            num_steps = total_time_avg / steptime_avg if steptime_avg > 0.0 else 0.0
            samptime_step = samptime_avg / num_steps if num_steps > 0.0 else 0.0
            with open(output_file, "a+") as f:
                data = f"PLANNER: {planners[i]} | costs: {np.average(distances[i]):0.2f} | "\
                        f"samptime_step: {samptime_step:0.2f} | steptime: {np.average(steptimes[i]):0.2f} | "\
                        f"num_steps: {num_steps:0.1f} | total runtime: {np.average(ttimes[i]):0.2f}"
                if prefix is not None:
                    data = data + prefix
                
                data = data + "\n"
                f.write(data)
    
    
    # if base_dist[0] != None:
    #     with open(output_file, "a+") as f:
    #         f.write(f"PLANNER: CTP        | UGVs: {args.num_ugvs} | AVG_COST: {np.average(base_dist):0.2f} |"
    #             f" AVG_STEP_TIME: {np.average(base_steptimes):0.2f} | SAMP_TIME: {np.average(base_samptimes):0.2f} |"
    #             f" T.TIME: {np.average(base_ttimes):0.2f} \n")    
   
    # if jsap_dist[0] != None:
    #     with open(output_file, "a+") as f:
    #         f.write(f"PLANNER: JSAP-1     | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsap_dist):0.2f} |"
    #             f" AVG_STEP_TIME: {np.average(jsap_steptimes):0.2f} | SAMP_TIME: {np.average(jsap_samptimes):0.2f} |"
    #             f" T.TIME: {np.average(jsap_ttimes):0.2f} \n")    
    
    # if jsap2_dist[0] != None:
    #     with open(output_file, "a+") as f:
    #         f.write(f"PLANNER: JSAP-2     | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsap2_dist):0.2f} |"
    #             f" AVG_STEP_TIME: {np.average(jsap2_steptimes):0.2f} | SAMP_TIME: {np.average(jsap2_samptimes):0.2f} |"
    #             f" T.TIME: {np.average(jsap2_ttimes):0.2f} \n")    
    
        
    # if jsapavp_dist[0] != None:
    #     with open(output_file, "a+") as f:
    #         data = f"PLANNER: JSAP-IAP-1 | UGVs: {args.num_ugvs} | AVG_COST: {np.average(jsapavp_dist):0.2f} |"\
    #             f" AVG_STEP_TIME: {np.average(jsapavp_steptimes):0.2f} | SAMP_TIME: {np.average(jsapavp_samptimes):0.2f} |"\
    #             f" T.TIME: {np.average(jsapavp_ttimes):0.2f}"
    #         if prefix is not None:
    #             data = data + prefix
    #         else:
    #             data = data + "\n"
    #         f.write(data)
    
def read_processed_data(filepath, prune_actions=False):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split('|')
            planner = parts[0].split(':')[1].strip()
            # ugvs = int(parts[1].split(':')[1].strip())
            # avg_cost = float(parts[2].split(':')[1].strip())
            # avg_step_time = float(parts[3].split(':')[1].strip())
            # samp_time = float(parts[4].split(':')[1].strip())
            # total_time = float(parts[5].split(':')[1].strip())
            # if len(parts) > 6:
            #     prune_num_action = int(parts[6].split(':')[1].strip())
                # print(f"The number of prune_action: {prune_num_action}")
            if planner not in data:
                data[planner] = {}
            
            if prune_actions:
        
                avg_cost = float(parts[1].split(':')[1].strip())
                samp_time = float(parts[2].split(':')[1].strip())
                avg_step_time = float(parts[3].split(':')[1].strip())
                avg_step_nums = float(parts[4].split(':')[1].strip())
                total_time = float(parts[5].split(':')[1].strip())
                prune_num_action = int(parts[6].split(':')[1].strip())
        
                assert len(parts) > 6, "Prune num action is specified but not found in the data"
                data[planner][prune_num_action] = {
                    'avg_cost': avg_cost,
                    'avg_step_time': avg_step_time,
                    'samp_time': samp_time,
                    'total_time': total_time,
                }
            else:
                ugvs = int(parts[1].split(':')[1].strip())
                avg_cost = float(parts[2].split(':')[1].strip())
                samp_time = float(parts[3].split(':')[1].strip())
                avg_step_time = float(parts[4].split(':')[1].strip())
                avg_step_nums = float(parts[5].split(':')[1].strip())
                total_time = float(parts[6].split(':')[1].strip())
                
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
        args.num_ugvs = 3
        input_path = Path(file_path)/ f'plot_data/results_{args.num_ugvs}UGVs.txt'
        args.save_dir = Path(args.save_dir)/ f'plot_data'
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
            
    elif prune_num_actions is not None:
        num_sampling = 5000
        output_filename = Path(file_path) / f'bridges/max_uav_action/processed_data_{num_sampling}.txt'
        # planner  = 'JSAP-IAP-1'
        args.num_ugvs = 1
        is_data_processed = True
        if not is_data_processed:
            for prune_num_action in prune_num_actions:
                input_filename = Path(file_path) / f'bridges/max_uav_action/1_{prune_num_action}_{num_sampling}/results_1UGVs.txt'
                prefix = f" | PRUNE_NUM_ACTION: {prune_num_action}"
                processed_data(input_file=input_filename, output_file=output_filename, args=args, prefix=prefix)
        
        data = read_processed_data(output_filename, prune_actions=True)
        distances = [[data[planner][prune_num_action]['avg_cost'] for prune_num_action in prune_num_actions] for planner in data]
        figure_out = Path(file_path) / f'bridges/max_uav_action/prune_actions_fig_{num_sampling}.pdf'
        plotting.plot_madist_allinOne_std(x=prune_num_actions, data=distances, std=None, featureNames=list(data.keys()), yName="Distances [m]",
                ranges=(200, 500), rangeStep=50, envName=f"Bridges_Graph", xName="number of candidate actions after pruning", outpath=figure_out)
        plt.show()
    