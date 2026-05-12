import numpy as np
import random, os
import glob
from sctp.learning.iap_gnn import BipartiteEdgeRegressor
# from sctp.learning.iap_gnn2 import EdgeIGGNN
from sctp.scripts.data_gen import generate_dataset, generate_dataset_by_rollout
from sctp.scripts.iapgnn_training import GzipGNNDataset, train_epoch
import pickle, gzip
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sctp.sctp_graphs as graphs
from sctp.robot import Robot
from sctp.param import RobotType
from sctp import jsap
from sctp.planners import jsap_planner
from sctp.planners import jsap_plan_exe as plan_loop
from sctp.utils import plotting
from sctp import action_estimation as ae
from sctp.learning.iap_gnn import load_iap_gnn_model
import time
import matplotlib.pyplot as plt

def test_data_generation():
    graph_type = 'bridges'
    sampling_nums = 500
    graph_nums = 10
    num_data_per_graph = 50
    filepath ='data/sctp/graph_data/'
    print(f"Graph_Type: {graph_type}, number of maps for sampling {sampling_nums}-number of graph: {graph_nums},"
          f" saving to: {filepath}") 
    
    generate_dataset(filepath=filepath, num_graphs=graph_nums, num_maps=sampling_nums, 
                        graph_type=graph_type, num_data_per_graph=num_data_per_graph)

def test_data_generation_rollout():
    graph_type = 'bridges'
    sampling_nums = 1000
    num_steps = 18
    seed = 1000
    filepath ='data/sctp/graph_data/rollouts/'
    print(f"Graph_Type: {graph_type}, number of maps for sampling {sampling_nums}, "
          f" saving to: {filepath}") 
    
    generate_dataset_by_rollout(filepath=filepath, seed=seed, num_maps=sampling_nums, 
                        graph_type=graph_type, num_steps=num_steps)



def test_GNN_model():
    # Hyperparameters
    NODE_IN = 2  # [Start, Goal]
    EDGE_IN = 2  # [Length, Prob] <-- Fixed: Changed from 3 to 2 to match data
    HIDDEN = 32

    model = BipartiteEdgeRegressor(NODE_IN, EDGE_IN, HIDDEN)

    # Dummy Data
    x = torch.tensor([[1,0], [0,0], [0,1], [0,0]], dtype=torch.float) # 4 Nodes
    # hot code to define start node and goal.
    
    # how about the Node we need to calculate the value?

    # Fixed: Transpose edge_index to [2, Num_Edges]
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).t().contiguous()
    # all connectivity between nodes (edges)
    
    edge_attr = torch.tensor([[5.0, 0.1], [4.0, 0.0], [4.0, 0.5]], dtype=torch.float)
    # [distance, prob]
    
    # Forward
    preds = model(x, edge_index, edge_attr)

    print(preds.shape) # Output: [3, 1] (One prediction per edge)
    print(preds)

def test_get_Graphdata():
    # Load the dataset
    with gzip.open('data/sctp/graph_data/pickles/dat_1000_1.pgz', 'rb') as f:
        data = pickle.load(f)
    print("Node Features (x):", data.x.shape)  # [Num_Nodes, Node_Feats]
    print("Node Features (x):", data.x)  # [Num_Nodes, Node_Feats]
    print("Edge Index:", data.edge_index.shape)  # [2, Num_Edges]
    print("Edge Index:", data.edge_index)  # [2, Num_Edges]
    print("Edge Attributes:", data.edge_attr.shape)  # [Num_Edges, Edge_Feats]
    print("Edge Attributes:", data.edge_attr)  # [Num_Edges, Edge_Feats]
    print("Target Values (y):", data.y.shape)  # [Num_Edges, 1]
    print("Target Values (y):", data.y)  # [Num_Edges, 1]

def test_IAPtraining():
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 126
    
    # Check the first data point
    all_files = glob.glob(os.path.join('data/sctp/graph_data/pickles/', '*.pgz'))
    dataset = GzipGNNDataset(all_files)
    # Use a DataLoader for batching
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BipartiteEdgeRegressor(NODE_IN, EDGE_IN, HIDDEN).to(device)
    # device = torch.device('cpu')  # Force CPU for debugging
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    print("Starting training...")
    num_epochs = 4000
    for epoch in range(1, num_epochs+1):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")
        
    
def test_IAPGNN_predictions():
    print()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    NODE_IN = 2
    EDGE_IN = 2
    HIDDEN = 64
    model = BipartiteEdgeRegressor(node_in_dim=NODE_IN, edge_in_dim=EDGE_IN, hidden_dim=HIDDEN).to(device)
    

    PATH = "/modules/sctp/learning/models/iap_gnn.pt" # The path to your saved model file

    # 3. Load the state dictionary
    # Use weights_only=True as a best practice
    model.load_state_dict(torch.load(PATH, weights_only=True))
    
    # test_model(model, dataset, device, n_samples=2)
    """
    Run inference on a random subset of graphs and print predicted vs
    ground-truth IG values side-by-side for every edge.

    Parameters
    ----------
    model      : trained EdgeIGGNN
    dataset    : list of PyG Data objects (each must have ig_labels)
    device     : torch device
    n_samples  : number of graphs to sample for inspection
    seed       : random seed for reproducible sampling
    """
    model.eval()
    # torch.manual_seed(seed)
    data_dir = 'data/sctp/graph_data/pickles/'
    all_files = glob.glob(os.path.join(data_dir, '*.pgz'))
    n_samples = 1
    dataset = GzipGNNDataset(all_files)

    indices = torch.randperm(len(dataset))[:n_samples].tolist()

    # ── aggregate metrics across all sampled graphs ────────────────────────
    all_pred    = []
    all_target  = []

    print("=" * 75)
    print(f"  MODEL TEST RESULTS  ({n_samples} randomly sampled graphs)")
    print("=" * 75)

    for sample_num, idx in enumerate(indices, 1):
        data   = dataset[idx].to(device)
        # target = prepare_ig_labels(data.ig_labels, data.edge_attr)

        with torch.no_grad():
            pred, _ = model(
                x          = data.x,
                edge_index = data.edge_index,
                edge_attr  = data.edge_attr,
            )   # [E]

        pred_cpu   = pred.cpu()
        target_cpu = data.y.cpu()
        E          = pred_cpu.shape[0]

        src = data.edge_index[0].cpu()   # [E]
        dst = data.edge_index[1].cpu()   # [E]
        p   = data.edge_attr[:, 1].cpu() # [E]

        uncertain_mask = (p > 0.0) & (p < 1.0)
        certain_mask   = ~uncertain_mask

        # ── per-graph header ───────────────────────────────────────────────
        print(f"\n{'─' * 75}")
        print(f"  Graph {sample_num}  (dataset index {idx})  |  "
              f"{E} edges  |  "
              f"{uncertain_mask.sum().item()} uncertain  |  "
              f"{certain_mask.sum().item()} certain")
        print(f"{'─' * 75}")
        print(f" {'Edge':>7}  {'p_block':>10}  {'Type':>8}  "
              f"{'GT-IG':>10}  {'Pred IG':>10}  {'Error':>10}  {'AbsErr':>9}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

        for i in range(E):
            u      = src[i].item()
            v      = dst[i].item()
            p_val  = p[i].item()
            gt     = target_cpu[i].item()
            pr     = pred_cpu[i].item()
            err    = pr - gt
            abs_err= abs(err)
            etype  = "uncertain" if uncertain_mask[i] else "certain"

            # flag large errors
            flag = " ◄" if abs_err > 1.0 and uncertain_mask[i] else ""

            print(f"  ({u:>2},{v:>2})     "
                  f"{p_val:>5.3f}  "
                  f"{etype:>10}  "
                  f"{gt:>9.4f}  "
                  f"{pr:>10.4f}  "
                  f"{err:>+11.4f}  "
                  f"{abs_err:>8.4f}"
                  f"{flag}")

            if uncertain_mask[i]:
                all_pred.append(pr)
                all_target.append(gt)

        # ── per-graph summary ──────────────────────────────────────────────
        unc_pred   = pred_cpu[uncertain_mask]
        unc_target = target_cpu[uncertain_mask]

        if uncertain_mask.any():
            mae  = (unc_pred - unc_target).abs().mean().item()
            rmse = ((unc_pred - unc_target) ** 2).mean().sqrt().item()
            bias = (unc_pred - unc_target).mean().item()

            # ranking accuracy: fraction of pairs ordered correctly
            dp = unc_pred.unsqueeze(0)   - unc_pred.unsqueeze(1)    # [K,K]
            dt = unc_target.unsqueeze(0) - unc_target.unsqueeze(1)  # [K,K]
            pairs = dt.abs() > 0.01
            rank_acc = ((dp * dt) > 0)[pairs].float().mean().item() if pairs.any() else float('nan')

            print(f"\n  Graph {sample_num} uncertain-edge metrics:")
            print(f"    MAE        = {mae:.4f}")
            print(f"    RMSE       = {rmse:.4f}")
            print(f"    Bias       = {bias:+.4f}  "
                  f"({'over-predicting' if bias > 0 else 'under-predicting'})")
            print(f"    Rank Acc   = {rank_acc:.3f}  "
                  f"(fraction of edge pairs ranked correctly)")
        else:
            print(f"\n  Graph {sample_num}: no uncertain edges to evaluate.")


def test_jsap_gnn_state():
    print()
    planner = 'jsapliap'
    seed = 3012
    random.seed(seed)
    np.random.seed(seed)
    num_ugvs = 1
    
    verbose = False
    # starts, goals, graph = graphs.get_sixIslands_graph()
    starts, goals, graph = graphs.random_graph(n_vertex=16,SG_pairs=3)
    # print("All the nodes in the graph with their block_prob:")
    # for v in graph.vertices:
    #     print(f"Node {v.id}: block_prob={v.block_prob}")
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    use_AVP=False
    use_DAP = False
    use_Learning = True
    num_drones = 1 
    max_depth = 15
    max_uanum = 2
    num_iterations = 1000
    n_maps = 200
    model_path = "/modules/sctp/learning/models/iap_gnn_allgraphs_m.pt"
    drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
            robot_type=RobotType.Drone, at_node=True) for i in range(num_drones)]
    print(f"Testing JSAP-IAP planner with use_IAP={use_AVP} and num_iterations={num_iterations} and max_depth={max_depth}")
    
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    # init_state = jsap.JSAPState(graph=policyGraph, goalIDs=[goal.id for goal in goals], n_maps=n_maps, \
    #                     revisit_pen=0.0, drones=drones, ugvs=robots, useAVP=use_AVP, useDAP=use_DAP, \
    #                     useLearning=use_Learning, max_uanum=max_uanum, model_path=model_path)
    
    jsapplanner = jsap_planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
                uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=200.0, revisit_pen=0.0, \
                rollout_num=num_iterations, tree_depth=max_depth, n_maps=n_maps, use_DAP=use_DAP, \
                use_AVP=use_AVP, useLearning=use_Learning, model_path=model_path, max_uanum=max_uanum, \
                verbose=True)
    
    
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_actions, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_actions, cost)
    
    robot_net_times = [robot.net_time for robot in robots]
    cost_sum = np.sum(robot_net_times)
    cost_aver = np.average(robot_net_times)
    
    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=False)
    plt.show()


def test_jsap_uavMaxActions():
    print()
    planner = 'jsapliap'
    seed = 3012
    random.seed(seed)
    np.random.seed(seed)
    num_ugvs = 2
    
    verbose = False
    # starts, goals, graph = graphs.get_sixIslands_graph()
    starts, goals, graph = graphs.random_graph(n_vertex=16,SG_pairs=3)
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    # plotting.plot_sctpgraph(graph, ax, verbose=True, initG=True)
    plotGraph = graph.copy()
    policyGraph = graph.copy()
    
    use_AVP=False
    use_DAP = False
    use_Learning = True
    num_drones = 1 
    max_depth = 25
    max_uanum = 1
    num_iterations = 10000
    n_maps = 200
    model_path = 'modules/sctp/learning/models/iap_gnn_allgraphs_200_May03.pt'
    # gnn_model = BipartiteEdgeRegressor(node_in_dim=2, edge_in_dim=2, hidden_dim=64).to('cpu')
    # gnn_cache = {}
    # if use_Learning:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     gnn_model = load_iap_gnn_model(path=model_path, device=device)
        
    drones = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
            robot_type=RobotType.Drone, at_node=True) for i in range(num_drones)]
    
    
    robots = [Robot(position=[starts[i].coord[0], starts[i].coord[1]], cur_node=starts[i].id, \
                    at_node=True) for i in range(num_ugvs)]
    planner_robots = [robot.copy() for robot in robots]
    planner_drones = [drone.copy() for drone in drones]
    
    # init_state = jsap.JSAPState(graph=policyGraph, goalIDs=[goal.id for goal in goals], n_maps=n_maps, \
    #                     revisit_pen=0.0, drones=drones, ugvs=robots, useAVP=use_AVP, useDAP=use_DAP, \
    #                     useLearning=use_Learning, max_uanum=max_uanum, gnn_model=gnn_model, device=device, gnn_cache=gnn_cache)
    
    # actions = ae.get_uav_action_gnn(init_state, 0)
    jsapplanner = jsap_planner.JSAPPlanner(init_graph=policyGraph, goalIDs=[goal.id for goal in goals], ugvs=planner_robots, \
                uavs=planner_drones, rollout_fn=jsap.decsctp_rollout, C=200.0, revisit_pen=0.0, \
                rollout_num=num_iterations, tree_depth=max_depth, n_maps=n_maps, use_DAP=use_DAP, \
                use_AVP=use_AVP, useLearning=use_Learning, model_path=model_path, max_uanum=max_uanum, \
                verbose=True)
    
    
    plan_exec = plan_loop.JSAPPlanExe(graph=graph, ugvs=robots, uavs=drones, goalIDs=[goal.id for goal in goals],\
                                                    reached_goal=jsapplanner.reached_goal, verbose=True)

    start_time = time.perf_counter() 
    average_step_time = 0.0
    count_steps = 0
    
    for step_data in plan_exec:
        jsapplanner.update(
            step_data['observed_pois'],
            step_data['ugvs'],
            step_data['uavs']
        )
        time1 = time.perf_counter()
        joint_actions, cost = jsapplanner.compute_joint_action()
        average_step_time += (time.perf_counter() - time1)
        count_steps += 1
        plan_exec.save_joint_actions(joint_actions, cost)
        # if count_steps > 2:
        break
    robot_net_times = [robot.net_time for robot in robots]
    cost_sum = np.sum(robot_net_times)
    cost_aver = np.average(robot_net_times)
    
    runtime = time.perf_counter() - start_time
    average_step_time /= count_steps
    gpaths = []
    for robot in robots:
        x_g = [pose[0] for pose in robot.all_poses]
        y_g = [pose[1] for pose in robot.all_poses]
        gpaths.append([x_g, y_g])
    
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    goals_cords = [goal.coord for goal in goals]
    starts_cords = [start.coord for start in starts]
    plotting.plot_plan_exec(graph=graph, plt=plt, name=planner, gpaths=gpaths, dpaths=dpaths, \
                    graph_plot=plotGraph, start_coords=starts_cords, goal_coords=goals_cords, \
                        seed=seed, cost=cost_sum, ttime=runtime, stime=average_step_time, verbose=False)
    plt.show()