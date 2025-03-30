import numpy as np
from sctp.utils import paths
from scipy.spatial import Delaunay, distance



class Graph():
    def __init__(self, vertices=[], edges=[]):
        self.vertices = vertices
        self.edges = edges
        self.pois = []

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

    def update(self, observations):
        for key, value in observations.items():
            pois = [poi for poi in self.pois if poi.id ==key]
            if pois:
                # pois[0].block_prob = float(value)
                pois[0].block_prob = float(pois[0].block_status)
    
    def copy(self):
        new_graph = Graph(vertices=self.vertices)
        new_graph.pois = [poi.copy() for poi in self.pois]
        new_graph.edges = self.edges
        return new_graph

class Vertex:
    _id_counter = 1
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
        


def linear_graph_unc():
    start_node = Vertex(coord=(0.0, 0.0))
    node1 = Vertex(coord=(5.0, 0.0))
    goal_node = Vertex(coord=(15.0, 0.0))
    nodes = [start_node, node1, goal_node]
    graph = Graph(nodes)
    graph.edges.clear()
    graph.add_edge(start_node, node1, 0.5)
    graph.add_edge(node1, goal_node, 0.3)
    paths.dijkstra(graph=graph, goal=goal_node)
    return start_node, goal_node, graph


def disjoint_unc():  # edge 34 is blocked
    # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
    nodes = []
    node1 = Vertex(coord=(0.0, 0.0))
    nodes.append(node1)
    node2 =  Vertex(coord=(4.0, 0.0))
    nodes.append(node2)
    node3 =  Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node3)
    node4 =  Vertex(coord=(4.0, 4.0))
    nodes.append(node4)

    graph = Graph(nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node3, node4, 0.1)
    graph.add_edge(node2, node3, 0.9)
    graph.add_edge(node1, node4, 0.2)
    # vertices = graph.vertices + graph.pois
    paths.dijkstra(graph=graph, goal=node3)
    return node1, node3, graph


def s_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = Vertex(coord=(4.0, 4.0))
    nodes.append(node2)
    node3 = Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node4)
    graph = Graph(nodes)
    graph.edges.clear()

    # adding edges
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node1, node3, 0.1)
    graph.add_edge(node2, node3, 0.1)
    graph.add_edge(node2, node4, 0.1)
    graph.add_edge(node3, node4, 0.9)
    paths.dijkstra(graph=graph, goal=node4)
    return node1, node4, graph


def m_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = Vertex(coord=(-3.0, 4.0)) # start node
    nodes.append(node1)
    node2 = Vertex(coord=(-4.0, 8.5))
    nodes.append(node2)
    node3 = Vertex(coord=(0.0, 2.0))
    nodes.append(node3)
    node4 = Vertex(coord=(4.0, 0.0))
    nodes.append(node4)
    node5 = Vertex(coord=(4.0, 4.0))
    nodes.append(node5)
    node6 = Vertex(coord=(4.0, 8.0))
    nodes.append(node6)
    node7 = Vertex(coord=(8.0, 4.0)) # goal node
    nodes.append(node7)

    graph = Graph(nodes)
    graph.edges.clear()

    # add edges
    graph.add_edge(node1, node2, 0.1) #8
    graph.add_edge(node1, node3, 0.1) #9
    graph.add_edge(node2, node3, 0.1) #10
    graph.add_edge(node2, node5, 0.1) #11
    graph.add_edge(node2, node6, 0.1)#12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node3, node5, 0.1) #14
    graph.add_edge(node4, node5, 0.1) #15
    graph.add_edge(node4, node7, 0.90) #16
    graph.add_edge(node5, node7, 0.90) #17
    graph.add_edge(node6, node7, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    paths.dijkstra(graph=graph, goal=node7)
    return node1, node7, graph



def generate_random_coordinates(n, xmin, ymin, xmax, ymax, min_dist, max_dist):
    points = []
    attempts = 0
    max_attempts = 5000
    while len(points) < n and attempts < max_attempts:
        point = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
        if not points:
            points.append(point)
            point = np.array([np.random.uniform(point[0]+1.5, point[0]+2.0), np.random.uniform(point[1]+1.5, point[1]+2.0)])
            points.append(point)
            continue
        dists = distance.cdist(np.array([point]), np.array(points)).flatten()
        if np.all(dists >= min_dist) and np.sum(dists <= max_dist) >= 2:
            points.append(point)
        attempts +=1
    
    if len(points) < n:
        raise ValueError("Cannot get enough vertices")
    return points


def random_graph(n_vertex=8, xmin=0, ymin=0):
    """Generate a random graph with Delaunay triangulation and weighted edges."""    
    size = 7.0 * (np.sqrt(n_vertex)-1.0)
    max_edge_length = 9.0
    min_edge_length = 3.0
    points = generate_random_coordinates(n_vertex, xmin=xmin, ymin=ymin, xmax=1.5*size, ymax=size,\
                                         min_dist=min_edge_length, max_dist=max_edge_length)
    tri = Delaunay(np.array(points))
    graph = Graph(vertices=[Vertex(coord=point) for point in points])
    graph.edges.clear()
    edge_count = {}

    # Use a set to avoid duplicate edges
    edges = set()    
    for simplex in tri.simplices:
        edges.update(tuple(sorted((simplex[i], simplex[j]))) for i in range(3) for j in range(i + 1, 3))    
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            edge = tuple(sorted((simplex[i], simplex[j])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    # Add edges to the graph with random weights
    for i, j in edges:
        dist = np.linalg.norm(np.array(graph.vertices[i].coord) - np.array(graph.vertices[j].coord))
        if dist > max_edge_length:
            continue
        if ((i, j) in boundary_edges or (j, i) in boundary_edges) and dist >max_edge_length-1.2:
            continue

        if np.random.random() <0.85: # control level of blockage in the graph
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.15, 0.6))
        else:
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.7, 0.90))
    startId = min(enumerate(points), key=lambda p: p[1][0])[0]
    start_pos = points[startId]
    goalId = max(enumerate(points), key=lambda p: np.linalg.norm(np.array(start_pos)- np.array(p[1])))[0]
    goal = graph.vertices[goalId]
    start = graph.vertices[startId]
    paths.dijkstra(graph=graph, goal=goal)
    trav_path = paths.get_random_path(graph, start=start.id, goal=goal.id)
    assert len(trav_path) > 1
    for vertex in graph.pois:
        if vertex.id in trav_path:
            vertex.block_status = int(0)
    return start, goal, graph
