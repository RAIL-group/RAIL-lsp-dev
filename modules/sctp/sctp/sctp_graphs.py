import numpy as np
from sctp.utils import paths
# from scipy.spatial import Delaunay, distance
from sctp.param import TRAV_LEVEL, MAX_EDGE_LENGTH, MIN_EDGE_LENGTH
from sctp import param
from sctp import graph as g
import math
MAX_ISLAND_DISTANCE = 35.0
MIN_ISLAND_DISTANCE = 20.0


def random_graph(n_vertex=14, xmin=0, ymin=0, SG_pairs=3):
    """Generate a random graph with Delaunay triangulation and weighted edges."""    
    count = 0
    while True:
        starts, goals, graph = g.generate_random_graph(n_vertex=n_vertex, xmin=xmin, ymin=ymin,\
                                max_edge_len=MAX_EDGE_LENGTH, min_edge_len=MIN_EDGE_LENGTH, num_sg=SG_pairs)
        start_goal_connected = True
        for i, start in enumerate(starts):
            if not g.check_graph_valid(startID=start.id, goalID=goals[i].id, graph=graph):
                start_goal_connected = False
                break
        if start_goal_connected:
            break
        
        count += 1
        if count > 15000:
            print("Cannot find a valid graph, try other seed ranges")
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return starts, goals, graph

def get_bridges_graph():
    count = 0
    while True:
        starts, goals, graph = create_bridges_graph()
        start_goal_connected = True
        for i, start in enumerate(starts):
            if not g.check_graph_valid(startID=start.id, goalID=goals[i].id, graph=graph):
                start_goal_connected = False
                break
        if start_goal_connected:
            break
        
        count += 1
        if count > 1000:
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return starts, goals, graph

def create_bridges_graph():
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(np.random.uniform(0.0,3.5), np.random.uniform(0.0,1.5))) # start node
    node2 = g.Vertex(coord=(np.random.uniform(0.0,3.0), np.random.uniform(22.0,27.5)))
    node3 = g.Vertex(coord=(np.random.uniform(0.0,3.5), np.random.uniform(48.0,55.5)))
    node4 = g.Vertex(coord=(np.random.uniform(20.0,23.5), np.random.uniform(11.0,16.5))) 
    node5 = g.Vertex(coord=(np.random.uniform(21.0,24.5), np.random.uniform(34.0,45.5)))
    node6 = g.Vertex(coord=(np.random.uniform(44.5,47.5), np.random.uniform(0.0,3.0)))
    node7 = g.Vertex(coord=(np.random.uniform(35.5,47.5), np.random.uniform(25.0,50.5)))
    node8 = g.Vertex(coord=(np.random.uniform(44.5,47.5), np.random.uniform(68.0,77.5)))
    node9 = g.Vertex(coord=(np.random.uniform(74.5,77.5), np.random.uniform(0.0,3.0)))
    node10 = g.Vertex(coord=(np.random.uniform(68.5,85.5), np.random.uniform(25.0,50.5)))
    node11 = g.Vertex(coord=(np.random.uniform(73.5,78.5), np.random.uniform(67.0,78.5)))
    node12 = g.Vertex(coord=(np.random.uniform(93.5,98.5), np.random.uniform(0.0,3.0)))
    node13 = g.Vertex(coord=(np.random.uniform(93.5,98.5), np.random.uniform(22.0,25.5)))
    node14 = g.Vertex(coord=(np.random.uniform(93.5,98.5), np.random.uniform(46.0,55.5)))
    node15 = g.Vertex(coord=(np.random.uniform(114.5,119.5), np.random.uniform(0.0, 4.5)))
    node16 = g.Vertex(coord=(np.random.uniform(113.5,118.5), np.random.uniform(24.0,30.5)))
    
    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, \
                    node9, node10, node11, node12, node13, node14, node15, node16]
    graph = g.Graph(nodes)
    graph.edges.clear()
    # initial_edges = []
    # bridges
    graph.add_edge(node6, node9, np.random.uniform(0.55, 0.75)) #17
    graph.add_edge(node7, node10, np.random.uniform(0.40, 0.55)) #18
    graph.add_edge(node8, node11, np.random.uniform(0.15, 0.3)) #19
    # start connections
    graph.add_edge(node1, node2, np.random.uniform(0.05,0.1)) #20
    graph.add_edge(node2, node3, np.random.uniform(0.05,0.1)) #21
    # start neighbors
    graph.add_edge(node1, node6, np.random.uniform(0.2,0.4)) #22
    graph.add_edge(node1, node4, np.random.uniform(0.2,0.4)) #23
    graph.add_edge(node2, node4, np.random.uniform(0.2,0.4)) #24
    graph.add_edge(node2, node5, np.random.uniform(0.2,0.4)) #25
    graph.add_edge(node3, node5, np.random.uniform(0.2,0.4)) #26
    graph.add_edge(node3, node8, np.random.uniform(0.2,0.4)) #27
    # goal connections
    graph.add_edge(node15, node16, np.random.uniform(0.05,0.1)) #28
    # goals neighbors
    graph.add_edge(node15, node12, np.random.uniform(0.2,0.4)) #29
    graph.add_edge(node15, node13, np.random.uniform(0.2,0.4)) #30
    graph.add_edge(node16, node13, np.random.uniform(0.2,0.4)) #31
    graph.add_edge(node16, node14, np.random.uniform(0.2,0.4)) #32
    # other edges
    graph.add_edge(node4, node6, np.random.uniform(0.2,0.7)) #33
    graph.add_edge(node4, node7, np.random.uniform(0.2,0.7)) #34
    graph.add_edge(node5, node7, np.random.uniform(0.2,0.7)) #35
    graph.add_edge(node5, node8, np.random.uniform(0.2,0.7)) #36
    graph.add_edge(node6, node7, np.random.uniform(0.2,0.7)) #37
    graph.add_edge(node7, node8, np.random.uniform(0.2,0.7)) #38
    graph.add_edge(node9, node10, np.random.uniform(0.2,0.7)) #39
    graph.add_edge(node10, node11, np.random.uniform(0.2,0.7)) #40
    graph.add_edge(node9, node12, np.random.uniform(0.2,0.7)) #41
    graph.add_edge(node9, node13, np.random.uniform(0.2,0.7))   #42
    graph.add_edge(node10, node13, np.random.uniform(0.2,0.70)) #43
    graph.add_edge(node10, node14, np.random.uniform(0.2,0.70)) #44
    graph.add_edge(node11, node14, np.random.uniform(0.2,0.70)) #45
    graph.add_edge(node12, node13, np.random.uniform(0.2,0.70)) #46
    graph.add_edge(node13, node14, np.random.uniform(0.2,0.70)) #47
    return [node1, node2, node3], [node15,node13, node16], graph

def random_island_graph(n_island=6, xmin=0, ymin=0, SG_dist_min=10):
    count = 0
    graph, islands, points = g.generate_island_graph(n_islands=n_island, xmin=xmin, ymin=ymin, 
                                max_edge_len=MAX_ISLAND_DISTANCE,min_edge_len=MIN_ISLAND_DISTANCE)
    while True:
        islands_vertices = [vertex for graph in islands for vertex in graph.vertices]
        start = min(enumerate(islands_vertices), key=lambda v: v[1].coord[0])[1]
        while True:
            goal = islands_vertices[np.random.randint(len(islands_vertices))]
            _, num_edges = paths.get_shortestPath_cost(graph=graph, start=start.id, goal=goal.id)
            if num_edges >=SG_dist_min:
                break
            
        if g.check_graph_valid(startID=start.id, goalID=goal.id, graph=graph):
            g.add_highways(out_graph=graph, islands=islands, points=points)
            break
        
        # reset the status of blocking point
        for poi in graph.pois:
            poi.block_status = int(0) if np.random.random() > poi.block_prob else int(1)
        
        count += 1
        if count > 2000:
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return start, goal, graph

def generate_islands_locations(xmin=0, ymin=0):
    locations = []
    loc1 = (np.random.uniform(xmin+0.0, xmin+5.0), np.random.uniform(ymin+0.0, ymin+5.0))
    locations.append(loc1)
    loc2 = (np.random.uniform(xmin+40.0, xmin+50.0), np.random.uniform(ymin+0.0, ymin+5.0))
    locations.append(loc2)
    loc3 = (np.random.uniform(xmin+95.0, xmin+105.0), np.random.uniform(ymin+0.0, ymin+10.0))
    locations.append(loc3)
    loc4 = (np.random.uniform(xmin+40.0, xmin+70.0), np.random.uniform(ymin+30.0, ymin+40.0))
    locations.append(loc4)
    loc5 = (np.random.uniform(xmin+5.0, xmin+15.0), np.random.uniform(ymin+55.0, ymin+65.0))
    locations.append(loc5)
    loc6 = (np.random.uniform(xmin+80.0, xmin+90.0), np.random.uniform(ymin+55.0, ymin+65.0))
    locations.append(loc6)
    return locations

def get_isolated_islands(locations):
    islands = []
    for i, loc in enumerate(locations):
        if i == 3:
            num_points = 5
        else:
            num_points = 4
        points = g.generate_points_around(loc, min_dist=5.5, max_dist=8.5, num_points=num_points)
        islands.append(points)
    return islands

def connect_inside_island(nodes, graph, island_id):
    for i in range(len(nodes)):
        if 0 < i < len(nodes):
            if island_id == 0 or island_id == 2:
                graph.add_edge(nodes[i], nodes[i-1], 0.0)
            else:
                graph.add_edge(nodes[i], nodes[i-1], np.random.uniform(0.1,0.2))        
    graph.add_edge(nodes[0], nodes[-1], 0.05)
    
def connect_island2island(graph):
    island1 = graph.vertices[0:4]
    island2 = graph.vertices[4:8]
    island3 = graph.vertices[8:12]
    island4 = graph.vertices[12:17]
    island5 = graph.vertices[17:21]
    island6 = graph.vertices[21:25]
    is1x_max = np.argmax([node.coord[0] for node in island1])
    is1y_max = np.argmax([node.coord[1] for node in island1])
    is1x_min = np.argmin([node.coord[0] for node in island1])
    is1y_min = np.argmin([node.coord[1] for node in island1])
    
    is2x_max = np.argmax([node.coord[0] for node in island2])
    is2x_min = np.argmin([node.coord[0] for node in island2])
    is2y_max = np.argmax([node.coord[1] for node in island2])
    is2y_min = np.argmin([node.coord[1] for node in island2])
    
    is3x_min = np.argmin([node.coord[0] for node in island3])
    is3x_max = np.argmax([node.coord[0] for node in island3])
    is3y_max = np.argmax([node.coord[1] for node in island3])
    is3y_min = np.argmin([node.coord[1] for node in island3])
    
    is4x_min = np.argmin([node.coord[0] for node in island4])
    is4y_min = np.argmin([node.coord[1] for node in island4])
    is4x_max = np.argmax([node.coord[0] for node in island4])
    is4y_max = np.argmax([node.coord[1] for node in island4])
    
    is5x_min = np.argmin([node.coord[0] for node in island5])
    is5y_min = np.argmin([node.coord[1] for node in island5])
    is5y_max = np.argmax([node.coord[1] for node in island5])
    
    is6x_min = np.argmin([node.coord[0] for node in island6])
    is6x_max = np.argmax([node.coord[0] for node in island6])
    is6y_max = np.argmax([node.coord[1] for node in island6])
    is6y_min = np.argmin([node.coord[1] for node in island6])
    
    # island 1 to island 2    
    graph.add_edge(island1[is1y_min], island2[is2x_min], np.random.uniform(0.5,0.6))
    # island 1 to island 4
    graph.add_edge(island1[is1y_max], island4[is4x_min], np.random.uniform(0.35,0.45))
    # island 1 to island 5
    graph.add_edge(island1[is1x_min], island5[is5x_min], np.random.uniform(0.15,0.3))
    # island 2 to island 3
    graph.add_edge(island2[is2y_min], island3[is3x_min], np.random.uniform(0.55,0.65))
    # island 2 to island 4
    graph.add_edge(island2[is2y_max], island4[is4y_min], np.random.uniform(0.35,0.45))
    # island 3 to island 4
    graph.add_edge(island3[is3y_max], island4[is4x_max], np.random.uniform(0.35,0.45))
    # island 3 to island 6
    graph.add_edge(island3[is3x_max], island6[is6x_max], np.random.uniform(0.15,0.3))
    # island 4 to island 5
    graph.add_edge(island4[is4y_max], island5[is5y_min], np.random.uniform(0.15,0.3))
    # island 4 to island 6
    graph.add_edge(island4[is4x_max], island6[is6x_min], np.random.uniform(0.15,0.3))
    # island 5 to island 6
    graph.add_edge(island5[is5y_max], island6[is6y_max], np.random.uniform(0.15,0.3))
    starts = [island1[is1x_min], island1[is1y_max], island1[is1y_min]]
    goals = [island3[is3x_max], island6[is6y_max], island3[is3y_min]]
    return starts, goals

def create_nodes(graph, island):
    nodes = []
    for i in range(len(island)):
        vertex = g.Vertex(coord=island[i])
        nodes.append(vertex)
        graph.add_vertex(vertex)
    return nodes

def connect_select_starts_goals(islands):
    g.Vertex.reset_id_counter()
    graph = g.Graph()
    graph.vertices.clear()
    graph.edges.clear()
    graph.pois.clear()
    graph.poiIDs.clear()
    islands_nodes = []
    for i, island in enumerate(islands):
        nodes = create_nodes(graph, island)
        islands_nodes.append(nodes)
        
    for i, nodes in enumerate(islands_nodes):
        connect_inside_island(nodes, graph, i)
    starts, goals = connect_island2island(graph)
    return starts, goals, graph   

def get_sixIslands_graph(xmin=0, ymin=0):
    locations = generate_islands_locations(xmin=xmin, ymin=ymin)
    
    count = 0
    while True:  
        islands = get_isolated_islands(locations)
        starts, goals, graph = connect_select_starts_goals(islands)
        start_goal_connected = True
        for i, start in enumerate(starts):
            if not g.check_graph_valid(startID=start.id, goalID=goals[i].id, graph=graph):
                start_goal_connected = False
                break
        if start_goal_connected:
            break
        
        count += 1
        if count > 2000:
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return starts, goals, graph

def random_bridges_graph(n_bridge=3):
    count = 0
    while True:
        start, goal, graph = create_3bridges_graph()
        if g.check_graph_valid(startID=start.id, goalID=goal.id, graph=graph):
            break
        count += 1
        if count > 10:
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return start, goal, graph

def create_3bridges_graph():
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(-10.0, 0.0)) # start node
    node2 = g.Vertex(coord=(np.random.uniform(8.5,11.5), np.random.uniform(-1.0,1.5)))
    node3 = g.Vertex(coord=(np.random.uniform(19.0,21.5), np.random.uniform(-1.0,1.5)))
    node4 = g.Vertex(coord=(np.random.uniform(29.0,31.5), np.random.uniform(-1.0,1.5))) # goal
    node5 = g.Vertex(coord=(np.random.uniform(9.0,11.5), np.random.uniform(29.0,31.5)))
    node6 = g.Vertex(coord=(np.random.uniform(19.0,21.5), np.random.uniform(29.0,31.5)))
    node7 = g.Vertex(coord=(np.random.uniform(9.0,11.5), np.random.uniform(-79.0,-81.5)))
    node8 = g.Vertex(coord=(np.random.uniform(19.0,21.5), np.random.uniform(-79.0,-81.5)))
    nodes = [node1, node2, node3, node4, node5, node6, node7, node8]
    graph = g.Graph(nodes)
    graph.edges.clear()
    rand1 = np.random.uniform(0.55,0.65)
    edge_rand  = np.random.randint(1,3)
    graph.add_edge(node1, node2, np.random.uniform(0.1,0.15))
    if edge_rand == 1:
        graph.add_edge(node2, node3, rand1)
    else:
        graph.add_edge(node2, node3, np.random.uniform(0.15,0.25))
    if edge_rand == 2:
        graph.add_edge(node3, node4, rand1)
    else:
        graph.add_edge(node3, node4, np.random.uniform(0.15,0.25))
    
    rand1 = np.random.uniform(0.4,0.5)
    edge_rand  = np.random.randint(1,3)
    graph.add_edge(node1, node5, np.random.uniform(0.1,0.15))
    if edge_rand == 1:
        graph.add_edge(node5, node6, rand1)
    else:
        graph.add_edge(node5, node6, np.random.uniform(0.1,0.2))
    if edge_rand == 2:
        graph.add_edge(node6, node4, rand1)
    else:
        graph.add_edge(node6, node4, np.random.uniform(0.1,0.2))
    graph.add_edge(node1, node7, np.random.uniform(0.05,0.1))
    graph.add_edge(node7, node8, np.random.uniform(0.05,0.1))
    graph.add_edge(node8, node4, np.random.uniform(0.05,0.1))
    return node1, node4, graph

def linear_graph_unc():
    g.Vertex.reset_id_counter()
    start_node = g.Vertex(coord=(0.0, 0.0))
    node1 = g.Vertex(coord=(5.0, 0.0))
    goal_node = g.Vertex(coord=(15.0, 0.0))
    nodes = [start_node, node1, goal_node]
    graph = g.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(start_node, node1, 0.5)
    graph.add_edge(node1, goal_node, 0.3)
    paths.dijkstra(graph=graph, goal=goal_node)
    return [start_node], [goal_node], graph


def disjoint_unc():  # edge 34 is blocked
    # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0))
    nodes.append(node1)
    node2 =  g.Vertex(coord=(4.0, 0.0))
    nodes.append(node2)
    node3 =  g.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node3)
    node4 =  g.Vertex(coord=(4.0, 4.0))
    nodes.append(node4)

    graph = g.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node3, node4, 0.1)
    graph.add_edge(node2, node3, 0.9)
    graph.add_edge(node1, node4, 0.2)
    # vertices = graph.vertices + graph.pois
    paths.dijkstra(graph=graph, goal=node3)
    return [node1], [node3], graph


def s_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 4.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node4)
    graph = g.Graph(nodes)
    graph.edges.clear()

    # adding edges
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node1, node3, 0.1)
    graph.add_edge(node2, node3, 0.1)
    graph.add_edge(node2, node4, 0.1)
    graph.add_edge(node3, node4, 0.9)
    paths.dijkstra(graph=graph, goal=node4)
    return [node1], [node4], graph

def s_graph_2goals():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 4.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, 4.0)) # goal node
    nodes.append(node5)
    graph = g.Graph(nodes)
    graph.edges.clear()

    # adding edges
    graph.add_edge(node1, node2, 0.1) # edge 1 - poi6 
    graph.add_edge(node1, node3, 0.1) # edge 2 - poi7
    graph.add_edge(node2, node3, 0.1) # edge 3 - poi8
    graph.add_edge(node2, node4, 0.1) # edge 4 - poi9
    graph.add_edge(node2, node5, 0.2) # edge 5 - poi10
    graph.add_edge(node3, node4, 0.9) # edge 6 - poi11
    graph.add_edge(node4, node5, 0.3) # edge 7 - poi12
    paths.dijkstra(graph=graph, goal=node4)
    return [node1, node1], [node4, node5], graph


def m_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(-4.0, 6.0)) # start node 1
    nodes.append(node1)
    node2 = g.Vertex(coord=(-3.0, 4.0)) # start node 2
    nodes.append(node2)
    node3 = g.Vertex(coord=(-2.0, 8.0)) # start node 3
    nodes.append(node3)
    node4 = g.Vertex(coord=(0.0, 2.0))
    nodes.append(node4)
    node5 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(4.0, 4.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(4.0, 7.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(7.0, 2.0)) # goal node 1
    nodes.append(node8) 
    node9 = g.Vertex(coord=(8.0, 0.0)) # goal node 2
    nodes.append(node9)
    node10 = g.Vertex(coord=(8.0, 4.0)) # goal node 3
    nodes.append(node10)
    

    graph = g.Graph(nodes)
    graph.edges.clear()

    # add edges
    graph.add_edge(node1, node2, 0.1) #11
    graph.add_edge(node1, node3, 0.1) #12
    graph.add_edge(node2, node3, 0.1) #13
    graph.add_edge(node2, node4, 0.1) #14
    graph.add_edge(node2, node6, 0.8) #15
    graph.add_edge(node3, node6, 0.1) #16
    graph.add_edge(node3, node7, 0.1) #17
    graph.add_edge(node4, node5, 0.1) #18
    graph.add_edge(node4, node6, 0.90) #19
    graph.add_edge(node5, node6, 0.20) #20
    graph.add_edge(node5, node8, 0.90) #21
    graph.add_edge(node5, node9, 0.90) #22
    graph.add_edge(node6, node7, 0.1) #23
    graph.add_edge(node6, node8, 0.1) #24
    graph.add_edge(node6, node10, 0.1) #25
    graph.add_edge(node7, node10, 0.1) #26
    graph.add_edge(node8, node9, 0.1) #27
    graph.add_edge(node8, node10, 0.1) #28
    graph.add_edge(node9, node10, 0.1) #29
    for poi in graph.pois:
        if poi.id in [11,12,13,27,28,29,16,25, 15, 24]:
            poi.block_status = 0
    return [node1, node2, node3], [node8, node9, node10], graph


def graph_stuck():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    # node2 = Vertex(coord=(0.2, 5.0))
    # node2 = Vertex(coord=(2.0, 2.5))
    node2 = g.Vertex(coord=(0.0, 2.5))
    nodes.append(node2)
    node3 = g.Vertex(coord=(8.0, 2.5))
    nodes.append(node3)
    node4 = g.Vertex(coord=(12.0, 1.0))
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node5)

    graph = g.Graph(nodes)
    graph.edges.clear()

    # add edges
    graph.add_edge(node1, node2, 0.25) #6
    graph.add_edge(node2, node3, 0.15) #7
    graph.add_edge(node3, node4, 0.84) #8
    # graph.add_edge(node3, node5, 0.88) #9
    graph.add_edge(node4, node5, 0.86) #10
    graph.add_edge(node1, node5, 0.77) #11
    paths.dijkstra(graph=graph, goal=node4)
    return node1, node4, graph


def island_sgraph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 3.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, -4.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, -4.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, 3.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(12.0, 0.0)) # goal node
    nodes.append(node8)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.2) #9
    graph.add_edge(node1, node3, 0.2) #10
    graph.add_edge(node1, node4, 0.1) #11
    graph.add_edge(node2, node3, 0.1) #12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node2, node7, 0.1) #14 - should be the edge with highest value
    graph.add_edge(node7, node8, 0.1) #15
    graph.add_edge(node6, node8, 0.1) #16
    graph.add_edge(node5, node8, 0.1) #17
    graph.add_edge(node7, node6, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    paths.dijkstra(graph=graph, goal=node8)
    return node1, node8, graph

def island_mgraph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 3.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 3.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, -4.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, -4.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, 3.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(12.0, 0.0)) # goal node
    nodes.append(node8)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.1) #9
    graph.add_edge(node1, node3, 0.1) #10
    graph.add_edge(node1, node4, 0.1) #11
    graph.add_edge(node2, node3, 0.1) #12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node2, node7, 0.5) #14 - should be the edge with highest value
    graph.add_edge(node7, node8, 0.4) #15
    graph.add_edge(node6, node8, 0.1) #16
    graph.add_edge(node5, node8, 0.1) #17
    graph.add_edge(node7, node6, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    graph.add_edge(node4, node5, 0.45) #20
    paths.dijkstra(graph=graph, goal=node8)
    return node1, node8, graph

def island_bridges_sgraph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    g.Vertex.reset_id_counter()
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(0.0, 20.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(20.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(20.0, 20.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(40.0, 0.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(40.0, 20.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(60.0, 0.0))
    nodes.append(node7)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.05) #8
    graph.add_edge(node1, node3, 0.2) #9
    graph.add_edge(node2, node3, 0.15) #10
    graph.add_edge(node2, node4, 0.15) #11 - could be the edge with highest value
    graph.add_edge(node3, node4, 0.2) #12
    graph.add_edge(node3, node5, 0.6) #13
    graph.add_edge(node4, node6, 0.25) #14
    graph.add_edge(node5, node6, 0.15) #15
    graph.add_edge(node5, node7, 0.08) #16
    graph.add_edge(node6, node7, 0.15) #17
    paths.dijkstra(graph=graph, goal=node7)
    return [node1], [node7], graph


