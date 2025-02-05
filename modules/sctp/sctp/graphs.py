import numpy as np
import random, copy
import matplotlib.pyplot as plt

class RobotData:
    def __init__(self, *,robot_id=None, position=None,
                    cur_node=None, last_robot=None):
        if last_robot is not None:
            self.robotID = last_robot.robotID
            if position is not None:
                self.position = position
            else:
                self.position = last_robot.position
            self.cur_vertex = last_robot.cur_vertex
            self.last_vertex = self.cur_vertex
        else:
            self.robotID = robot_id
            if position is not None:
                self.position = position
            else:
                self.position = [cur_node.coord[0], cur_node.coord[1]] 
            self.cur_vertex = cur_node.id
            self.last_vertex = None

   # def __eq__(self, other):
   #    return self == other.__hash__()
   
   # def __hash_(self):
   #    return hash(self.robotID
    def getID(self):
        return self.robotID

class Graph():
    def __init__(self, vertices=[], edges=[]):
        self.vertices = vertices
        self.edges = edges

    def add_vertex(self, vertex):
        if isinstance(vertex, list):
            self.vertices.extend(vertex)
        else:
            self.vertices.append(vertex)

    def add_edge(self, vertex1, vertex2, block_prob=0.0):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Vertices not in graph. Add vertices before adding edges.")
        edge = Edge(vertex1, vertex2, block_prob)
        self.edges.append(edge)
        vertex1.neighbors.append(vertex2.id)
        vertex2.neighbors.append(vertex1.id)


class Vertex:
    _id_counter = 1
    def __init__(self, coord):
        self.id = Vertex._id_counter
        Vertex._id_counter += 1
        self.parent = None
        self.coord = coord
        self.neighbors = []

    def get_id(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id) + hash(str(self.coord))


class Edge:
    def __init__(self, v1, v2, block_prob=0.0):
        self.v1 = v1
        self.v2 = v2
        self.hash_id = self.__hash__()
        self.dist = np.linalg.norm(
            np.array((v1.coord[0], v1.coord[1])) - np.array((v2.coord[0], v2.coord[1])))
        self.cost = self.dist
        self.poi_coord = None
        self.block_prob = block_prob
        self.block_status = 1 if random.random() < block_prob else 0

    def get_cost(self) -> float:
        return self.cost

    def update_cost(self, value):
        self.cost = value

    def get_poi_coord(self):
        return self.poi_coord

    def get_blocked_prob(self):
        return self.block_prob

    def __eq__(self, other):
        return self.hash_id == other.hash_id

    def __hash__(self):
        return hash(self.v1) + hash(self.v2)


def orientation(p, q, r):
    """Return orientation of ordered triplet (p, q, r).
    0 -> collinear, 1 -> clockwise, 2 -> counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def do_segments_intersect(nodep1, nodep2, nodeq1, nodeq2):
    """Return True if line segments p1p2 and q1q2 intersect."""
    p1 = [nodep1.x, nodep1.y]
    p2 = [nodep2.x, nodep2.y]
    p_middle = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    p1[0] = 0.5*(p1[0]+p_middle[0])
    p1[1] = 0.5*(p1[1]+p_middle[1])

    p2[0] = 0.5*(p2[0]+p_middle[0])
    p2[1] = 0.5*(p2[1]+p_middle[1])
    q1 = [nodeq1.x, nodeq1.y]
    q2 = [nodeq2.x, nodeq2.y]

    q_middle = [(q1[0] + q2[0]) / 2, (q1[1] + q2[1]) / 2]
    q1[0] = 0.5*(q1[0]+q_middle[0])
    q1[1] = 0.5*(q1[1]+q_middle[1])
    q2[0] = 0.5*(q2[0]+q_middle[0])
    q2[1] = 0.5*(q2[1]+q_middle[1])
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special case: p1, p2, q1, q2 are collinear, and q lies on segment p
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def generate_street_graph(grid_rows, grid_cols, edge_probability):
    """Generate a graph resembling a street system."""
    nodes = []
    edges = []

    # Create nodes at grid intersections
    node_id = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Nodes aligned in a grid pattern
            x, y = 5.0*j + \
                random.uniform(-1.0, 1.0), 5.0*i + random.uniform(-1.0, 1.0)
            nodes.append(Vertex(node_id, (x, y)))
            node_id += 1

    # Connect nodes with edges (including diagonals)
    for node in nodes:
        neighbors = []

        # Horizontal and vertical neighbors
        if node.id % grid_cols < grid_cols - 1:  # Right neighbor
            neighbors.append(nodes[node.id + 1])
        if node.id + grid_cols < len(nodes):  # Bottom neighbor
            neighbors.append(nodes[node.id + grid_cols])

        for neighbor in neighbors:
            if random.random() <= edge_probability:
                new_edge = Edge(node, neighbor, random.uniform(0, 1))
                if all(not do_segments_intersect(new_edge.v1, new_edge.v2, edge.v1, edge.v2) for edge in edges):
                    edges.append(new_edge)

        neighbors = []
        # Diagonal neighbors
        if node.id % grid_cols < grid_cols - 1 and node.id + grid_cols < len(nodes):  # Bottom-right neighbor
            neighbors.append(nodes[node.id + grid_cols + 1])
        if node.id % grid_cols > 0 and node.id + grid_cols < len(nodes):  # Bottom-left neighbor
            neighbors.append(nodes[node.id + grid_cols - 1])

        # Add edges while ensuring no intersections
        for neighbor in neighbors:
            if random.random() < 0.55:
                new_edge = Edge(node, neighbor, random.uniform(0, 1))
                if all(not do_segments_intersect(new_edge.v1, new_edge.v2, edge.v1, edge.v2) for edge in edges):
                    edges.append(new_edge)

    return nodes, edges


def m_graph_unc():
   """Generate a simple graph for testing purposes."""
   nodes = []
   node1 = Vertex(coord=(-3.0, 4.0)) # start node
   nodes.append(node1)
   node2 = Vertex(coord=(-15.0, 7.5))
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
   
   # add edges
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node1, node3, 0.1)
   graph.add_edge(node2, node3, 0.1)
   graph.add_edge(node2, node5, 0.1)
   graph.add_edge(node2, node6, 0.1)
   graph.add_edge(node3, node4, 0.1)
   graph.add_edge(node3, node5, 0.1)
   graph.add_edge(node4, node5, 0.1)
   graph.add_edge(node4, node7, 0.1)
   graph.add_edge(node5, node7, 0.1)
   graph.add_edge(node6, node7, 0.1)
   graph.add_edge(node6, node5, 0.1)

   robots = RobotData(robot_id=1, position=[-3.0, 4.0], cur_node=node1)
   return node1, node7, graph, robots


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

   # adding edges
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node1, node3, 0.1)
   graph.add_edge(node2, node3, 0.1)
   graph.add_edge(node2, node4, 0.1)
   graph.add_edge(node3, node4, 0.95)

   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node1)
   return node1, node4, graph, robots


def disjoint_unc():  # edge 34 is blocked

   # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
   nodes = []
   node1 = Vertex(coord=(0.0, 0.0))
   nodes.append(node1)
   node2 =  Vertex(coord=(4.0, 0.0))
   nodes.append(node2)
   node3 =  Vertex(coord=(8.0, 0.0))
   nodes.append(node3)
   node4 =  Vertex(coord=(4.0, 4.0))
   nodes.append(node4)

   graph = Graph(nodes)
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node3, node4, 0.1)
   graph.add_edge(node2, node3, 0.9)
   graph.add_edge(node1, node4, 0.2)
   node_new = copy.copy(node1)
   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node_new)
   return node1, node3, graph, robots


def linear_graph_unc():
   nodes = []
   node1 = Vertex(coord=(0.0, 0.0))
   nodes.append(node1)
   node2 = Vertex(coord=(5.0, 0.0))
   nodes.append(node2)
   node3 = Vertex(coord=(15.0, 0.0))
   nodes.append(node3)
   graph = Graph(nodes)
   graph.add_edge(node1, node2, 0.9)
   graph.add_edge(node2, node3, 0.0)
   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node1)
   return node1, node3, graph, robots


# helper functions
def plot_street_graph(nodes, edges, name="Testing Graph"):
    """Plot graph using matplotlib."""
    plt.figure(figsize=(10, 10))

    # Plot edges
    for edge in edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]
        plt.plot(x_values, y_values, 'b-', alpha=0.7)
        # Display block probability
        mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
        mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
        probs = f"{edge.block_prob:.2f}/" + f"{edge.block_status}/{edge.cost:.1f}"
        plt.text(mid_x, mid_y, probs, color='red', fontsize=8)

    # Plot nodes
    for node in nodes:
        plt.scatter(node.coord[0], node.coord[1], color='black', s=50)
        plt.text(node.coord[0] + 0.1, node.coord[1] + 0.1, f"{node.id}", color='blue', fontsize=8)

    plt.title(name)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    plt.show()


def print_graph(nodes, edges, show_edge=False, show_node=False):
    """Print the graph details."""
    if show_node:
        for node in nodes:
            print(f"Node {node.id}: ({node.coord[0]:.2f}, {node.coord[1]:.2f}) with neighbors: {node.neighbors}")
            print(f"The neighbors features:")
            for n in node.neighbors:
                edge = [edge for edge in edges if ((edge.v1.id == node.id and edge.v2.id == n) or (edge.v1.id == n and edge.v2.id == node.id))][0]
                print(f"edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")
    if show_edge:
        for edge in edges:
            print(f"Edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")

if __name__ == "__main__":
    # Parameters for the street system
    grid_rows = 5  # Number of rows in the grid
    grid_cols = 5  # Number of columns in the grid
    edge_probability = 1.0  # Probability of creating an edge between nodes

    # Generate and plot the street-like graph
    # nodes, edges = generate_street_graph(grid_rows, grid_cols, edge_probability)
    # name = "Street Graph"
    # nodes, edges = simple_graph()
    # name = "Simple Graph"
    # nodes, edges = simple_disjoint_graph()
    # name = "Simple Disjoint Graph"
    # print_graph(nodes, edges, show_edge=True)
    # print("THe neighbors are:")
    # for node in nodes:
    #     print(f"Node {node.id}: with neighbors: {node.neighbors}")

    # plot_street_graph(nodes, edges, name)
