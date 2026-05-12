import os
import random, argparse
import numpy as np
# import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
import sctp.sctp_graphs as graphs
import sctp.utils.underlying_graph as ug
import sctp.action_estimation as ae
from torch_geometric.data import Data
import torch
import learning
from sctp.learning.iap_gnn import load_iap_gnn_model
from sctp.planners import jsap_planner as planner 
from sctp.planners import jsap_plan_exe as plan_loop
from sctp.robot import Robot
from sctp import graph as g
from sctp import jsap, core
from sctp.param import VEL_RATIO, RobotType


@dataclass
class GraphData:
    """Data structure to store graph information"""
    x: np.ndarray #start, goal
    edge_index: np.ndarray #edge_index
    edge_attr: np.ndarray #edge_attr
    y: np.ndarray #y [M ,1]
    graph_metadata: Dict

def graphdata_to_pyg(sdata: GraphData, device='cpu') -> Data:
    """Convert GraphData to PyTorch Geometric Data object"""
    data = Data(
        x          = torch.tensor(sdata.x, dtype=torch.float32),
        edge_index = torch.tensor(sdata.edge_index, dtype=torch.long),   # shape [2, E]
        edge_attr  = torch.tensor(sdata.edge_attr, dtype=torch.float32),
        y          = torch.tensor(sdata.y, dtype=torch.float32),
    )
    return data.to(device)

def create_graph_datum(
    graph: ug.ProbabilisticGraph,
    edges: List,
    start: int, # 0-indexed
    goal: int, # 0-indexed
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> GraphData:
    # 1. Create node features
    # 1. Create node features [Num_Nodes, 2]
    num_nodes = graph.adjacency.shape[0]
    nodes = np.zeros((num_nodes, 2), dtype=np.float32)
    # Mark start and goal using one-hot style features
    nodes[start, 0] = 1.0
    nodes[goal, 1] = 1.0
    
    edge_attr = []
    for u, v in edges:
        length = graph.adjacency[u, v]
        prob = graph.probabilities[u, v]
        edge_attr.append([length, prob])
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'num_nodes': num_nodes,
        'num_edges': len(edges),
    })
    
    return GraphData(
        x= nodes,
        edge_index=np.array(edges, dtype=np.int64).T,
        edge_attr=np.array(edge_attr, dtype=np.float32),
        y=np.array(values, dtype=np.float32).reshape(-1, 1),
        graph_metadata=metadata,
    )


def generate_dataset(
    filepath: str,
    seed: int,
    num_maps: int = 500,
    graph_type: str = 'bridges',
    num_data_per_graph: int = 50,
    verbose=False
):
    """
    Generate a dataset of multiple graphs
    Args:
        num_graphs: Number of graphs to generate
        graph_type: Type of graph ('random', 'bridges', 'islands')
    
    Returns:
        List of GraphData objects
    """
    assert num_maps == 1000
    # seeds = np.arange(1021, 1021+num_graphs)
    # for i in range(num_graphs):
    np.random.seed(seed)
    random.seed(seed)
    # Generate random number of nodes
    if graph_type == 'islands':
        _, _, graph = graphs.get_sixIslands_graph()
    elif graph_type == 'random':
        assert args.n_vertex == 16, f"Number of vertices for random graph should be at least 10, got {args.n_vertex}"
        _, _, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=3)
    elif graph_type == 'bridges':
        _, _, graph = graphs.get_bridges_graph()
    else:
        raise ValueError(f"Graph type {graph_type} not recognized")
    edges = ug.get_initial_edges(graph)
    vertex_positions = ug.get_vertex_positions(graph.vertices)
    adjacency_matrix, probability_matrix = ug.create_adj_prob_matrices(edges, vertex_positions)
    count = 0
    if verbose:
        file_summary = os.path.join('pickles_new', f'dat_{graph_type}_{seed}.txt')
        file_summary = os.path.join(filepath, file_summary)
    
    vertices_id = [vertex.id for vertex in graph.vertices]
    for ii in range(len(vertices_id)-1):
        assert vertices_id[ii] < vertices_id[ii+1], f"Vertices are not in the correct order: {vertices_id}"
        
    while count < num_data_per_graph:
        while True:
            start, goal = random.sample(range(0, len(graph.vertices)), 2)
            dist = np.linalg.norm(np.array(graph.vertices[start].coord) - np.array(graph.vertices[goal].coord))
            if dist > 25.0: # ensure start and goal are not too close
                break
        num_known_edges = random.randint(0, 8)
        assert num_known_edges <= len(graph.pois), f"Number of known edges {num_known_edges} cannot exceed total number of edges {len(graph.pois)}"
        known_edges = random.sample(graph.pois, num_known_edges)
        known_edge_probs = [0.0 if random.random() >= poi.block_prob else 1.0 for poi in known_edges]
        known_edges_id = [[poi.neighbors[0]-1, poi.neighbors[1]-1] for poi in known_edges] #0-index
        new_probability_matrix = ug.set_edge_probabilities(
            probs=np.array(known_edge_probs),
            edges=known_edges_id,
            probabilities=probability_matrix
        )
        result = new_probability_matrix[np.isin(new_probability_matrix, [0.0, 1.0])]
        assert result.size > 5, f"Not enough known edges: {result.size} found, expected at least 5"
        pg = ug.ProbabilisticGraph(
            positions=vertex_positions,
            adjacency=adjacency_matrix,
            probabilities=new_probability_matrix
        )
        values = []
        edge_list = []
        for poi in graph.pois:
            edge = [poi.neighbors[0]-1, poi.neighbors[1]-1] # calculate its value
            assert edge[0] < edge[1], f"Edge {edge} is not in the correct order"
            edge_list.append(edge)
            if edge in known_edges_id:
                values.append(0.0)
                continue    
            
            bc = ae.get_single_bc_networkX(
                ugraph=pg,
                action_edge=[edge],
                start=start,
                goalID=goal,
                n_samples=num_maps
            )
            values.append(bc) if bc >= 0.0 else values.append(0.0)
        assert len(edge_list) == len(values), f"Number of edges {len(edge_list)} does not match number of values {len(values)}"
        graph_data = create_graph_datum(graph=pg, edges=edge_list, start=start, goal=goal, values=values)
        write_datum_to_file(filepath=filepath, seed=seed, datum=graph_data, counter=count, graph_type=graph_type)
        count += 1
        if verbose:
            with open(file_summary, "a+") as f:
                f.write(f"START: {start} | GOAL: {goal}\n")
                f.write(f"EDGE_LIST: {edge_list}\n")
                f.write(f"VALUES: {values}\n")

def generate_dataset_by_rollout(
    filepath: str,
    seed: int,
    num_maps: int = 1000,
    graph_type: str = 'bridges',
    num_steps: int = 20,
    verbose=True
):
    """
    Generate a dataset of multiple graph by rollout
    Args:
        num_graphs: Number of graphs to generate
        graph_type: Type of graph ('random', 'bridges', 'islands')
    
    Returns:
        List of GraphData objects
    """
    assert num_maps == 1000
    assert graph_type == 'random'
    C = 200.0
    num_iterations = 1000
    max_depth = 12
    # verbose = True
    
    # model_path = 'modules/sctp/learning/models/iap_gnn_allgraphs_60k.pt'
    model_path = 'modules/sctp/learning/models/iap_gnn_allgraphs_200_May03.pt'
    np.random.seed(seed)
    random.seed(seed)
    # Generate random number of nodes
    if graph_type == 'islands':
        starts, goals, graph = graphs.get_sixIslands_graph()
    elif graph_type == 'random':
        assert args.n_vertex == 16, f"Number of vertices for random graph should be at least 10, got {args.n_vertex}"
        starts, goals, graph = graphs.random_graph(n_vertex=args.n_vertex, SG_pairs=3)
    elif graph_type == 'bridges':
        starts, goals, graph = graphs.get_bridges_graph()
    else:
        raise ValueError(f"Graph type {graph_type} not recognized")
    filepath = os.path.join(filepath, f'{graph_type}/')
    edges = ug.get_initial_edges(graph)
    vertex_positions = ug.get_vertex_positions(graph.vertices)
    adjacency_matrix, probability_matrix = ug.create_adj_prob_matrices(edges, vertex_positions)
    
    if verbose:
        file_summary = os.path.join(filepath, f'dat_{graph_type}_{seed}.txt')
        # file_summary = os.path.join(filepath, file_summary)
    
    vertices_id = [vertex.id for vertex in graph.vertices]
    for ii in range(len(vertices_id)-1):
        assert vertices_id[ii] < vertices_id[ii+1], f"Vertices are not in the correct order: {vertices_id}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_iap_gnn_model(path=model_path, device=device)
    count_all = 0
    for ii, goal in enumerate(goals):
        start = starts[ii]
        if not g.check_graph_valid(startID=start.id, goalID=goal.id, graph=graph):
            continue
        
        # drones = []
        step = 0
        robots = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)]
        planner_robots = [robot.copy() for robot in robots]
        inde_drone = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True, robot_type=RobotType.Drone)
        observed_graph = graph.copy()
        edges = ug.get_initial_edges(observed_graph)
        
        jsapplanner = planner.JSAPPlanner(init_graph=observed_graph, goalIDs=[goal.id], ugvs=planner_robots, 
                                                uavs=[], rollout_fn=jsap.decsctp_rollout, C=C, 
                                                rollout_num=num_iterations, tree_depth=max_depth, n_maps=1000, 
                                                use_AVP=False, use_DAP=False, useLearning=False, model_path=model_path,\
                                                max_uanum=1, verbose=False)
    
        known_pois = [] #random.sample(graph.pois, num_known_edges)
        known_edge_probs = []
        known_edges_id = [] #0-index
        
        while step < num_steps and not jsapplanner.reached_goal():
            # print(f"Step {step} | Goal: {goal.id} | Known POIS: {known_pois}")
            new_probability_matrix = ug.set_edge_probabilities(
                probs=np.array(known_edge_probs),
                edges=known_edges_id,
                probabilities=probability_matrix
            )
            pg = ug.ProbabilisticGraph(
                positions=vertex_positions,
                adjacency=adjacency_matrix,
                probabilities=new_probability_matrix
            )
            values = []
            edge_list = []
            # find the starting point of the ground robot
            ugv = jsapplanner.ugvs[0]
            if ugv.at_node:
                if ugv.last_node in observed_graph.poiIDs:
                    node = observed_graph.get_vertex_by_id(ugv.last_node)
                    e = [node.neighbors[0], node.neighbors[1]] # calculate its value
                    if node.block_prob == 1.0:
                        cur_node = ugv.pl_vertex
                    else:
                        cur_node = e[1] if e[0] == ugv.pl_vertex else e[0]
                else:
                    cur_node = ugv.last_node
            else:
                cur_node = ugv.edge[0] if ugv.edge[1] in observed_graph.poiIDs else ugv.edge[1]
            # print(f"Approximate node for ground robot: {cur_node}")
            for poi in observed_graph.pois:
                edge = [poi.neighbors[0]-1, poi.neighbors[1]-1] # calculate its value
                assert edge[0] < edge[1], f"Edge {edge} is not in the correct order"
                edge_list.append(edge)
                if edge in known_edges_id:
                    values.append(0.0)
                    continue    
                
                bc = ae.get_single_bc_networkX(
                    ugraph=pg,
                    action_edge=[edge],
                    start=cur_node-1,
                    goalID=goal.id-1,
                    n_samples=num_maps
                )
                values.append(bc) if bc >= 0.0 else values.append(0.0)
            assert len(edge_list) == len(values), f"Number of edges {len(edge_list)} does not match number of values {len(values)}"          
            graph_data = create_graph_datum(graph=pg, edges=edge_list, start=cur_node-1, goal=goal.id-1, values=values)
            write_datum_to_file(filepath=filepath, seed=seed, datum=graph_data, counter=count_all, graph_type=graph_type)
            if verbose:
                with open(file_summary, "a+") as f:
                    f.write(f"START: {cur_node} | GOAL: {goal.id}\n")
                    f.write(f"EDGE_LIST: {edge_list}\n")
                    f.write(f"VALUES: {values}\n")
            
            # ground robot
            actions, costs = jsapplanner.compute_joint_action()            
            end_pos = observed_graph.get_vertex_by_id(actions[0].target).coord
            distance = np.linalg.norm(np.array(ugv.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - ugv.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            ugv.retarget(actions[0], distance, direction)
            # drone
            avail_actions = []
            for poi in observed_graph.pois:
                if poi.block_prob == 0.0 or poi.block_prob == 1.0:
                    continue
                avail_actions.append(core.Action(target=poi.id))
            drone_action = ae.get_bestAction_gnn(edges=edges, graph = jsapplanner.observed_graph, startID=cur_node, goalID= goal.id,\
                            device=device, gnn_model=model, drone_pose=inde_drone.cur_pose, actions=avail_actions)
            end_pos = observed_graph.get_vertex_by_id(drone_action.target).coord
            distance = np.linalg.norm(np.array(inde_drone.cur_pose) - np.array(end_pos))
            direction = (np.array([end_pos[0], end_pos[1]]) - inde_drone.cur_pose)/distance if distance != 0.0 else np.array([1.0, 1.0])
            
            inde_drone.retarget(drone_action, distance=distance, direction=direction)
            
            # move both robots and update the observed graph
            min_time = min(ugv.remaining_time, inde_drone.remaining_time)
            ugv.advance_time(min_time)
            ugv.remaining_time = 0.0
            inde_drone.advance_time(min_time)
            inde_drone.remaining_time = 0.0
            if ugv.at_node:
                if ugv.last_node == goal.id:
                    break
                if ugv.last_node in observed_graph.poiIDs:
                    known_poi = observed_graph.get_vertex_by_id(ugv.last_node)
                    if 0.0 <known_poi.block_prob < 1.0:
                        status = 0.0 if known_poi.block_status == 0 else 1.0
                        known_poi.block_prob = status
                        known_pois.append(known_poi)
                        known_edge_probs.append(0.0 if known_poi.block_status == 0 else 1.0)
                        known_edges_id.append([known_poi.neighbors[0]-1, known_poi.neighbors[1]-1]) #0-index
            
            if inde_drone.at_node:
                if inde_drone.last_node != goal.id:
                    known_poi = observed_graph.get_vertex_by_id(inde_drone.last_node)
                    if 0.0 <known_poi.block_prob < 1.0:
                        status = 0.0 if known_poi.block_status == 0 else 1.0
                        known_poi.block_prob = status
                        known_pois.append(known_poi)
                        known_edge_probs.append(0.0 if known_poi.block_status == 0 else 1.0)
                        known_edges_id.append([known_poi.neighbors[0]-1, known_poi.neighbors[1]-1]) #0-index
            step += 1
            count_all += 1
            


def write_datum_to_file(filepath, seed, datum, counter, graph_type='bridges'):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    # data_filename = os.path.join('pickles_new', f'dat_{graph_type}_{seed}_{counter}.pgz')
    # learning.data.write_compressed_pickle(os.path.join(filepath, data_filename), datum)
    data_filename = os.path.join(filepath, f'dat_{graph_type}_{seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(data_filename, datum)
    
    csv_filename = f'graph_data_address_new.csv'
    with open(os.path.join(filepath, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')
        

def _setup(args):
    print(f"Graph_Type: {args.graph_type}, number of maps: {args.num_maps}-with seed: {args.seed},"
          f" saving to: {args.save_dir}") 
    verbose = False
    # generate_dataset(filepath=args.save_dir, seed=args.seed, num_maps=args.num_maps, \
    #                     graph_type=args.graph_type, verbose=verbose)
    generate_dataset_by_rollout(filepath=args.save_dir, seed=args.seed, num_maps=args.num_maps, \
                        graph_type=args.graph_type, num_steps=args.num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--num_maps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--graph_type', type=str, default='bridges')
    parser.add_argument('--n_vertex', type=int, default=16, help='Number of vertices for random graph')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of steps for rollout data generation')
    args = parser.parse_args()

    _setup(args)
