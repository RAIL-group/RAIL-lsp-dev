import numpy as np
from sctp.utils import paths, plotting
from scipy.spatial import Delaunay, distance
from sctp.param import TRAV_LEVEL, MAX_EDGE_LENGTH, MIN_EDGE_LENGTH
import math, random



class Graph():
    def __init__(self, vertices=[], edges=[]):
        self.vertices = vertices
        self.edges = edges
        self.pois = []
        self.poiIDs = []

    def add_vertex(self, vertex):
        if isinstance(vertex, list):
            self.vertices.extend(vertex)
        else:
            self.vertices.append(vertex)

    def add_edge(self, vertex1, vertex2, block_prob=0.0):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Vertices not in graph. Add vertices before adding edges.")
        poi_coord = (0.5*(vertex1.coord[0]+vertex2.coord[0]),0.5*(vertex1.coord[1]+vertex2.coord[1]))
        POI = Vertex(coord=poi_coord, block_prob=block_prob)
        self.pois.append(POI)
        self.poiIDs.append(POI.id)
        rand_cost = np.random.randint(1,10)

        edge1 = Edge(vertex1, POI)
        self.edges.append(edge1)
        edge1.rand_cost = rand_cost
        vertex1.neighbors.append(POI.id)
        POI.neighbors.append(vertex1.id)

        edge2 = Edge(POI, vertex2)
        edge2.rand_cost = rand_cost
        self.edges.append(edge2)
        POI.neighbors.append(vertex2.id)
        vertex2.neighbors.append(POI.id)
    
    def get_edge(self, id1, id2):
        for edge in self.edges:
            if (edge.v1.id == id1 and edge.v2.id == id2) or (edge.v1.id == id2 and edge.v2.id == id1):
                return edge
        raise ValueError("Edge not found in graph.")
    
    def get_poi(self, poi_id):
        if poi_id not in self.poiIDs:
            print(f"The poiID: {poi_id} is not in the list of poi's IDs: {self.poiIDs}")
            raise ValueError("POI ID not found in graph.")
        indx = self.poiIDs.index(poi_id)
        # print("The poi's ID: ", indx)
        return self.pois[self.poiIDs.index(poi_id)]
    
    def get_vertex_by_id(self, vertex_id):
        for vertex in self.vertices:
            if vertex.id == vertex_id:
                return vertex
        raise ValueError("Vertex not found in graph.")

    def update(self, observations):
        for key, value in observations.items():
            pois = [poi for poi in self.pois if poi.id ==key]
            for poi in pois:
                poi.block_prob = float(poi.block_status)
            # if pois:
                # pois[0].block_prob = float(value)
    
    def copy(self):
        new_vertices = [vertex.copy() for vertex in self.vertices]
        new_pois = [poi.copy() for poi in self.pois]
        edges = []
        for edge in self.edges:
            vs = [v for v in new_vertices+new_pois if v.id == edge.v1.id or v.id == edge.v2.id]
            edges.append(Edge(vs[0], vs[1]))
        # new_graph = Graph(vertices=new_vertices, edges=self.edges.copy())
        new_graph = Graph(vertices=new_vertices, edges=edges)
        new_graph.pois = new_pois
        new_graph.poiIDs = self.poiIDs.copy()
        return new_graph
    
    def print_graph_config(self):
        for v in self.vertices:
            print("Vertex ID: {}, Coord: ({:.2f},{:.2f}), Neighbors: {}".format(v.id, v.coord[0], v.coord[1], v.neighbors))
        for p in self.pois:
            print("POI ID: {}, Coord: ({:.2f},{:.2f}), Neighbors: {}, Block Prob: {:.2f}, Block Status: {}".format(\
                p.id, p.coord[0], p.coord[1], p.neighbors, p.block_prob, p.block_status))
        for e in self.edges:
            print("Edge between Vertex {} and Vertex {}, Cost: {:.2f}".format(\
                e.v1.id, e.v2.id, e.cost))
        

class Vertex:
    _id_counter = 1
    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 1
        
    def __init__(self, coord, block_prob=float(0.0)):
        self.id = Vertex._id_counter
        Vertex._id_counter += 1
        self.coord = coord
        self.neighbors = []
        self.heur2goal = 0.0
        self.block_prob = block_prob
        if self.block_prob == 0.0:
            self.block_status = int(0)
        else:
            self.block_status = int(0) if np.random.random() > block_prob else int(1)
    def get_id(self):
        return self.id
    def copy(self):
        new_vertex = Vertex(coord=(self.coord[0], self.coord[1]), block_prob=self.block_prob)
        new_vertex.id = self.id
        new_vertex.neighbors = self.neighbors.copy()
        new_vertex.heur2goal = self.heur2goal 
        new_vertex.block_status = self.block_status
        new_vertex.block_prob = self.block_prob
        return new_vertex
    def __eq__(self, other):
        return self.id == other.id
    def __hash__(self):
        return hash(self.id) + hash(str(self.coord))


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.hash_id = self.__hash__()
        self.dist = np.linalg.norm(
            np.array((v1.coord[0], v1.coord[1])) - np.array((v2.coord[0], v2.coord[1])))
        self.cost = self.dist
        self.ran_cost = 0
    def get_cost(self) -> float:
        return self.cost
    def __eq__(self, other):
        return self.hash_id == other.hash_id

    def __hash__(self):
        return hash(self.v1) + hash(self.v2)
    

def generate_random_coordinates(n, xmin, ymin, xmax, ymax, min_dist, max_dist):
    points = []
    attempts = 0
    max_attempts = 20000
    while len(points) < n and attempts < max_attempts:
        point = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
        if not points:
            points.append(point)
            point = np.array([np.random.uniform(point[0]+min_dist-4.5, point[0]+min_dist), 
                              np.random.uniform(point[1]+min_dist-4.5, point[1]+min_dist)])
            points.append(point)
            continue
        dists = distance.cdist(np.array([point]), np.array(points)).flatten()
        if np.all(dists >= min_dist) and np.sum(dists <= max_dist) >= 2:
            points.append(point)
        attempts +=1
    
    if len(points) < n:
        raise ValueError("Cannot get enough random points")
    return points

def calculate_triangle_angles(points, triangle):
    """Calculate the three angles (in degrees) of a triangle given its vertices."""
    A, B, C = points[triangle]
    a = np.linalg.norm(B - C)  # Opposite vertex A
    b = np.linalg.norm(A - C)  # Opposite vertex B
    c = np.linalg.norm(A - B)  # Opposite vertex C
    
    def angle_cos(a, b, c):
        cos_val = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_val = max(min(cos_val, 1.0), -1.0)  # Clamp for numerical stability
        return math.degrees(math.acos(cos_val))
    
    angle_A = angle_cos(a, b, c)
    angle_B = angle_cos(b, a, c)
    angle_C = angle_cos(c, a, b)
    
    return angle_A, angle_B, angle_C

def generate_random_graph(n_vertex, xmin, ymin, max_edge_len, min_edge_len, num_sg=1):
    size = 0.5*(max_edge_len+min_edge_len) * (np.sqrt(n_vertex))
    angle_min = 10.0
    points = generate_random_coordinates(n_vertex, xmin=xmin, ymin=ymin, xmax=1.8*size, ymax=size,\
                                         min_dist=min_edge_len, max_dist=max_edge_len)
    tri = Delaunay(np.array(points))
    valid_triangles = []
    for simplex in tri.simplices:
        angles = calculate_triangle_angles(np.array(points), simplex)
        if min(angles) > angle_min:
            valid_triangles.append(simplex)
    Vertex.reset_id_counter()
    graph = Graph(vertices=[Vertex(coord=point) for point in points])
    graph.edges.clear()
    edge_count = {}
    # Use a set to avoid duplicate edges
    edges = set()    
    for simplex in valid_triangles:
        edges.update(tuple(sorted((simplex[i], simplex[j]))) for i in range(3) for j in range(i + 1, 3))    
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            edge = tuple(sorted((simplex[i], simplex[j])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    # Add edges to the graph with random weights
    for i, j in edges:
        dist = np.linalg.norm(np.array(graph.vertices[i].coord) - np.array(graph.vertices[j].coord))
        if dist > max_edge_len:
            continue
        if ((i, j) in boundary_edges or (j, i) in boundary_edges) and dist >max_edge_len-1.2:
            continue
        if np.random.random() <TRAV_LEVEL: # control level of blockage in the graph
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.1, 0.6))
        else:
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.6, 0.80))
    # prune edges if they are too many
    max_edges = n_vertex * 2
    while len(graph.pois) > max_edges:
        poi_to_remove = random.choice(graph.pois)
        neighbor1 = graph.get_vertex_by_id(poi_to_remove.neighbors[0])
        neighbor2 = graph.get_vertex_by_id(poi_to_remove.neighbors[1])
        if len(neighbor1.neighbors) >=4 and len(neighbor2.neighbors) >=4:
            graph.poiIDs.remove(poi_to_remove.id)
            graph.pois.remove(poi_to_remove)
            graph.edges = [edge for edge in graph.edges if edge.v1.id != poi_to_remove.id and edge.v2.id != poi_to_remove.id]
            neighbor1.neighbors.remove(poi_to_remove.id)
            neighbor2.neighbors.remove(poi_to_remove.id)
        
    startId = min(enumerate(points), key=lambda p: p[1][0])[0]
    start_pos = points[startId]
    goalId = max(enumerate(points), key=lambda p: np.linalg.norm(np.array(start_pos)- np.array(p[1])))[0]
    goal = graph.vertices[goalId]
    start = graph.vertices[startId]
    starts = get_n_neighbors(num_sg, start, graph)
    goals = get_n_neighbors(num_sg, goal, graph)
    return starts, goals, graph

def get_neighbor_vertices(node, graph):
    neighbors = []
    poi_neighs = node.neighbors
    for poi_id in poi_neighs:
        v_neis = graph.get_poi(poi_id)
        v_neis.block_prob = 0.0
        v_neis.block_status = 0
        neighs = v_neis.neighbors
        
        assert len(neighs) ==2
        if neighs[0] != node.id:
            neighbors.append(graph.get_vertex_by_id(neighs[0]))
        else:
            neighbors.append(graph.get_vertex_by_id(neighs[1]))            
    return neighbors


def get_n_neighbors(n, node, graph):
    n_neighbors = [node]
    count = 0
    while len(n_neighbors) < n:
        new_neighbors = get_neighbor_vertices(n_neighbors[count], graph)
        for nei in new_neighbors:
            if nei not in n_neighbors:
                n_neighbors.append(nei)
                if len(n_neighbors) >= n:
                    break
        count +=1
    return n_neighbors
        

def generate_island_graph(xmin, ymin, max_edge_len, min_edge_len,n_islands=10):
    size = (max_edge_len+min_edge_len) * (np.sqrt(n_islands))
    points = generate_random_coordinates(n_islands, xmin=xmin, ymin=ymin, xmax=1.5*size, ymax=size,\
                                         min_dist=min_edge_len, max_dist=max_edge_len)
    out_graph, islands = generate_islands(points, min_dist=4.0, max_dist=5.0)    
    return out_graph, islands, points

def generate_islands(points, min_dist, max_dist):
    local_trav_level = 0.5
    graph_vertices = []
    graph_edges = []
    graph_pois = []
    graphs = []
    Vertex.reset_id_counter()
    for point in points:
        ps = generate_points_around(point, min_dist=min_dist, max_dist=max_dist, num_points=5)
        graph = Graph(vertices=[Vertex(coord=coord) for coord in ps+[point]]) 
        graph.edges.clear()
        tri = Delaunay(np.array(ps+[point]))
        edges = set()    
        for simplex in tri.simplices:
            edges.update(tuple(sorted((simplex[i], simplex[j]))) for i in range(3) for j in range(i + 1, 3))            
        for i, j in edges:
            graph.add_edge(graph.vertices[i], graph.vertices[j], block_prob=local_trav_level)
        graph_vertices.extend(graph.vertices)
        graph_edges.extend(graph.edges)
        graph_pois.extend(graph.pois)
        graphs.append(graph)    
    out_graph = Graph(vertices=graph_vertices, edges=graph_edges)
    out_graph.pois = graph_pois
    
    # adding bridges
    angle_min = 30.0
    tri = Delaunay(np.array(points))
    valid_triangles = []
    for simplex in tri.simplices:
        angles = calculate_triangle_angles(np.array(points), simplex)
        if min(angles) > angle_min:
            valid_triangles.append(simplex)
        edges = set()    
        for simplex in valid_triangles:
            edges.update(tuple(sorted((simplex[i], simplex[j]))) for i in range(3) for j in range(i + 1, 3))    
    for i, j in edges:
        node1, node2 = connect_islands(graphs[i], graphs[j])
        if node1 is None or node2 is None:
            raise ValueError("Cannot connect islands, no valid nodes found.")
        out_graph.add_edge(node1, node2, block_prob=local_trav_level)
    # add_highways(out_graph, graphs, points)
    return out_graph, graphs

def generate_points_around(point, min_dist, max_dist, num_points=5):
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    distances = np.random.uniform(min_dist, max_dist, num_points)
    points = []
    for angle, dist in zip(angles, distances):
        x = point[0] + dist * np.cos(angle)
        y = point[1] + dist * np.sin(angle)
        points.append((x, y))
    return points

def connect_islands(graph1, graph2):
    # Find the closest vertices between two graphs
    min_dist = float('inf')
    closest_pair = (None, None)
    
    for v1 in graph1.vertices:
        for v2 in graph2.vertices:
            dist = np.linalg.norm(np.array(v1.coord) - np.array(v2.coord))
            if dist < min_dist:
                min_dist = dist
                closest_pair = (v1, v2)    
    return closest_pair[0], closest_pair[1]

def add_highways(out_graph, islands, points):
    graph_vertices = [vertex for graph in islands for vertex in graph.vertices]
    x_max = max(enumerate(graph_vertices), key=lambda v: v[1].coord[0])[1].coord[0] + 2.0
    x_min = min(enumerate(graph_vertices), key=lambda v: v[1].coord[0])[1].coord[0] - 2.0
    y_max = max(enumerate(graph_vertices), key=lambda v: v[1].coord[1])[1].coord[1] + 2.0
    y_min = min(enumerate(graph_vertices), key=lambda v: v[1].coord[1])[1].coord[1] - 2.0
    
    cornerlu = Vertex(coord=(x_min, y_max))
    out_graph.vertices.append(cornerlu)
    cornerld = Vertex(coord=(x_min, y_min))
    out_graph.vertices.append(cornerld)
    cornerru = Vertex(coord=(x_max, y_max))
    out_graph.vertices.append(cornerru)
    cornerrd = Vertex(coord=(x_max, y_min))
    out_graph.vertices.append(cornerrd)
    closest_edges = []
    for point in points:
        cledge = closest_edge(x_min, y_min, x_max, y_max, point)
        closest_edges.append(cledge)
    left_vertices = []
    right_vertices = []
    top_vertices = []
    bottom_vertices = []
    for i, graph in enumerate(islands):
        cledge = closest_edges[i]
        if cledge == 'left':
            node = min(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1]
            if node.coord[0] < x_min + 5.0:
                left_vertices.append(Vertex(coord=(x_min, node.coord[1])))
        elif cledge == 'right':
            node = max(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1]
            if node.coord[0] > x_max -5.0:
                right_vertices.append(Vertex(coord=(x_max, node.coord[1])))
        elif cledge == 'top':
            node = max(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1]
            if node.coord[1] > y_max -5.0:
                top_vertices.append(Vertex(coord=(node.coord[0], y_max)))
        else:
            node = min(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1]
            if node.coord[1] < y_min +5.0:
                bottom_vertices.append(Vertex(coord=(node.coord[0], y_min)))
    out_graph.vertices.extend(left_vertices)
    out_graph.vertices.extend(right_vertices)
    out_graph.vertices.extend(top_vertices)
    out_graph.vertices.extend(bottom_vertices)
    # sorting vertices along the left, right, bottom, top edges
    left_vertices.append(cornerlu)
    left_vertices.append(cornerld)
    right_vertices.append(cornerru)
    right_vertices.append(cornerrd)
    top_vertices.append(cornerlu)
    top_vertices.append(cornerru)
    bottom_vertices.append(cornerld)
    bottom_vertices.append(cornerrd)
    left_vertices = sorted(left_vertices, key=lambda v: v.coord[1])
    right_vertices = sorted(right_vertices, key=lambda v: v.coord[1])
    bottom_vertices = sorted(bottom_vertices, key=lambda v: v.coord[0])
    top_vertices = sorted(top_vertices, key=lambda v: v.coord[0])
    for i in range(len(left_vertices)-1):
        out_graph.add_edge(left_vertices[i], left_vertices[i+1], block_prob=0.01)
        if i > 0 and i < len(left_vertices)-1:
            closest_node = min(enumerate(graph_vertices), key=lambda v: np.linalg.norm(np.array(left_vertices[i].coord)-np.array(v[1].coord)))[1]
            out_graph.add_edge(left_vertices[i], closest_node, block_prob=0.01)
    for i in range(len(right_vertices)-1):
        out_graph.add_edge(right_vertices[i], right_vertices[i+1], block_prob=0.01)
        if i > 0 and i < len(right_vertices)-1:
            closest_node = min(enumerate(graph_vertices), key=lambda v: np.linalg.norm(np.array(right_vertices[i].coord)-np.array(v[1].coord)))[1]
            out_graph.add_edge(right_vertices[i], closest_node, block_prob=0.01)
    for i in range(len(top_vertices)-1):
        out_graph.add_edge(top_vertices[i], top_vertices[i+1], block_prob=0.01)
        if i > 0 and i < len(top_vertices)-1:
            closest_node = min(enumerate(graph_vertices), key=lambda v: np.linalg.norm(np.array(top_vertices[i].coord)-np.array(v[1].coord)))[1]
            out_graph.add_edge(top_vertices[i], closest_node, block_prob=0.01)
    for i in range(len(bottom_vertices)-1):
        out_graph.add_edge(bottom_vertices[i], bottom_vertices[i+1], block_prob=0.01)
        if i > 0 and i < len(bottom_vertices)-1:
            closest_node = min(enumerate(graph_vertices), key=lambda v: np.linalg.norm(np.array(bottom_vertices[i].coord)-np.array(v[1].coord)))[1]
            out_graph.add_edge(bottom_vertices[i], closest_node, block_prob=0.01)
    return out_graph
    
        
def closest_edge(xmin, ymin, xmax, ymax, point):
    if not (xmin < point[0] < xmax and ymin < point[1] < ymax):
        raise ValueError("Point must be inside the rectangle")
    
    # Calculate distances to each edge
    distances = {
        'left': abs(point[0] - xmin),
        'right': abs(xmax - point[0]),
        'bottom': abs(point[1] - ymin),
        'top': abs(ymax - point[1])
    }
    
    # Return the edge with the smallest distance
    return min(distances, key=distances.get)

def check_graph_valid(startID, goalID, graph):
    block_pois = []
    for poi in graph.pois:
        if poi.block_status ==1:
            block_pois.append(poi.id)
    new_graph = remove_pois(graph=graph, poiIDs=block_pois)
    return paths.is_reachable(graph=new_graph, start=startID, goal=goalID)

def remove_blockEdges(graph):
    graph_copy = graph.copy()
    block_pois = []
    for poi in graph_copy.pois:
        if poi.block_prob == 1.0:
            block_pois.append(poi.id)
    graph_copy.edges = [edge for edge in graph_copy.edges \
                    if edge.v1.id not in block_pois and edge.v2.id not in block_pois]
    graph_copy.pois = [poi for poi in graph_copy.pois if poi.id not in block_pois]
    for vertex in graph_copy.vertices:
        vertex.neighbors = [nei for nei in vertex.neighbors if nei not in block_pois]
    return graph_copy

def remove_edge(graph, redge =[]):
    graph_copy = graph.copy()
    graph_copy.edges = [edge for edge in graph_copy.edges \
                    if not (edge.v1.id in redge and edge.v2.id in redge)]
    for vertex in graph_copy.vertices+graph_copy.pois:
        if vertex.id in redge:  
            vertex.neighbors = [nei for nei in vertex.neighbors if nei not in redge]
    return graph_copy

def remove_poi(graph, poiID):
    graph_copy = graph.copy() #remove_blockEdges(graph=graph)
    graph_copy.pois = [poi for poi in graph_copy.pois if poi.id!=poiID]
    graph_copy.edges = [edge for edge in graph_copy.edges if edge.v1.id!=poiID and edge.v2.id!=poiID]
    count = 0
    for vertex in graph_copy.vertices:
        if poiID in vertex.neighbors:
            count += 1
            vertex.neighbors = [nei for nei in vertex.neighbors if nei != poiID]
            if count >=2:
                break
    return graph_copy

def remove_pois(graph, poiIDs=[]):
    if not poiIDs:
        return graph.copy()
    graph_copy = graph.copy() #remove_blockEdges(graph=graph)
    graph_copy.pois = [poi for poi in graph_copy.pois if poi.id not in poiIDs]
    graph_copy.edges = [edge for edge in graph_copy.edges if edge.v1.id not in poiIDs and edge.v2.id not in poiIDs]
    for vertex in graph_copy.vertices:
        # if any(poi in poiIDs for poi in vertex.neighbors):
        vertex.neighbors = [nei for nei in vertex.neighbors if nei not in poiIDs]
    return graph_copy

def modify_graph(graph, robot_edge, poiIDs=[]):
    new_poiIDs = [poiID for poiID in poiIDs if poiID not in robot_edge]
    new_graph = remove_pois(graph=graph, poiIDs=new_poiIDs)
    if len(poiIDs) == len(new_poiIDs): # if GV is not on a removed POI
        return new_graph 
    else:
        poi_robot = [poi for poi in robot_edge if poi in poiIDs][0]
        other_side = robot_edge[0] if poi_robot == robot_edge[1] else robot_edge[1]
        p = [poi for poi in graph.pois if poi.id == poi_robot][0]
        block_side = p.neighbors[0] if p.neighbors[1] == other_side else p.neighbors[1]
        return remove_edges(new_graph, redges=[[block_side, other_side]])

def modify_graph_multiDrones(graph, robot_edges, poiIDs=[]):
    edge_pois = set([poi for edge in robot_edges for poi in edge])
    new_poiIDs = [poiID for poiID in poiIDs if poiID not in edge_pois]
    new_graph = remove_pois(graph=graph, poiIDs=new_poiIDs)
    if len(poiIDs) == len(new_poiIDs): # if GV is not on a removed POI
        return new_graph 
    else:
        edges = []
        for edge in robot_edges:
            poi_robot = [poi for poi in edge if poi in poiIDs][0]
            other_side = edge[0] if poi_robot == edge[1] else edge[1]
            p = [poi for poi in graph.pois if poi.id == poi_robot][0]
            block_side = p.neighbors[0] if p.neighbors[1] == other_side else p.neighbors[1]
            edges.append([block_side, other_side])
        
        return remove_edges(new_graph, redges=edges)


def remove_edges(graph, redges=None):
    if not redges:
        return graph.copy()
    graph_copy = graph.copy()
    vertex_map = {vertex.id: vertex for vertex in graph_copy.vertices+graph_copy.pois}
    edges_to_keep = []
    for edge in graph_copy.edges:
        if not any((edge.v1.id == e[0] and edge.v2.id == e[1]) or (edge.v1.id == e[1] and edge.v2.id == e[0]) for e in redges):
            edges_to_keep.append(edge)
    graph_copy.edges = edges_to_keep
    
    for e in redges:
        v1, v2 = vertex_map.get(e[0]), vertex_map.get(e[1])
        assert v1 and v2
        v1.neighbors = [nei for nei in v1.neighbors if nei != e[1]]
        v2.neighbors = [nei for nei in v2.neighbors if nei != e[0]]
    return graph_copy

def get_poi_value(graph, poiID, startID, goalID):
    graphw = remove_blockEdges(graph)
    sp_wpoi = paths.get_shortestPath_cost(graphw, start=startID, goal=goalID)
    if sp_wpoi < 0.0:
        return -1.0  # No way to goal on with this graph, should not take this action
    graphwo = remove_poi(graphw, poiID)
    sp_wopoi = paths.get_shortestPath_cost(graphwo, start=startID, goal=goalID)
    if sp_wopoi<0.0: # without this edge/action, no way to goal - should check first
        return 10.0
    return sp_wopoi - sp_wpoi


def get_removedEdges_allrobots(graph, robots, block_pois):
    vertices_connectRobot = set()
    edges = [] # those edges to be removed
    for robot in robots:
        if robot.at_node == True:
            vertices_connectRobot.add(robot.last_node)
            if robot.last_node in block_pois:
                neighbors = [node for node in graph.pois if node.id == robot.last_node][0].neighbors
                other_side = [n for n in neighbors if n != robot.pl_vertex][0]
                edges.append([robot.last_node, other_side])
        else:
            vertices_connectRobot.add(robot.edge[0])
            vertices_connectRobot.add(robot.edge[1])
            if robot.edge[0] in block_pois:
                pois = [node for node in graph.pois if node.id == robot.edge[0]] #robot.last_node]
                if pois != []:
                    neighbors = pois[0].neighbors
                    other_side = [n for n in neighbors if n != robot.edge[1]]
                    if other_side is not None:
                        edges.append([robot.edge[0], other_side[0]])
            if robot.edge[1] in block_pois:
                neighbors = [node for node in graph.pois if node.id == robot.edge[1]][0].neighbors
                other_side = [n for n in neighbors if n != robot.edge[0]]
                if other_side is not None:
                    edges.append([robot.edge[1], other_side[0]])
        
    # be sure no robot on the edges to be removed
    for robot in robots:
        if not robot.at_node:
            if robot.edge in edges:
                edges = [e for e in edges if e != robot.edge]
            if [robot.edge[1], robot.edge[0]] in edges:
                edges = [e for e in edges if e != [robot.edge[1], robot.edge[0]]]
    return edges, vertices_connectRobot
