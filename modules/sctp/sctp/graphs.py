import numpy as np
import random
import matplotlib.pyplot as plt
   
class RobotData:
   def __init__(self, *,robot_id=None, position=None, 
                cur_vertex=None, last_robot=None):
      if last_robot is not None:
         self.robotID = last_robot.robotID
         self.position = last_robot.position
         self.cur_vertex = last_robot.cur_vertex
         self.last_vertex = self.cur_vertex
      else:
         self.robotID = robot_id
         self.position = position
         self.cur_vertex = cur_vertex
         self.last_vertex = None

   def get_robotID(self) -> int:
      return self.robotID

class Vertex:
   def __init__(self, id: int, coord):
      self.id = id
      self.parent = None
      self.coord = coord
    #   self.y = y
      self.neighbors = []
   
   def get_id(self):
      return self.id

class Edge:
   def __init__(self, v1, v2, block_prob=0.0):
      self.id = tuple(sorted((v1.get_id(), v2.get_id())))
      self.v1 = v1
      self.v2 = v2
      
      self.dist = np.linalg.norm(np.array((v1.coord[0], v1.coord[1])) - np.array((v2.coord[0], v2.coord[1])))
      self.cost = self.dist
      self.poi_coord = None
      self.block_prob = block_prob
      self.block_status = 1 if random.random() < block_prob else 0
   
   def get_cost(self) -> float:
    
      return self.cost

   def update_cost(self, value):
      self.cost = value
   
   def get_edge_id(self):
      return self.id
   
   def get_poi_coord(self):
      return self.poi_coord
   
   def get_blocked_prob(self):
      return self.block_prob


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
            x, y = 5.0*j + random.uniform(-1.0,1.0), 5.0*i + random.uniform(-1.0,1.0)  # Nodes aligned in a grid pattern
            nodes.append( Vertex(node_id, (x, y)))
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
                new_edge =  Edge(node, neighbor, random.uniform(0, 1))
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
                new_edge =  Edge(node, neighbor, random.uniform(0, 1))
                if all(not do_segments_intersect(new_edge.v1, new_edge.v2, edge.v1, edge.v2) for edge in edges):
                    edges.append(new_edge)

    return nodes, edges

def m_graph_unc():
    """Generate a simple graph for testing purposes."""
    start = 1
    goal = 7
    nodes = []
    node1 = Vertex(1, (-3.0, 4.0))
    nodes.append(node1)
    node2 = Vertex(2, (-15.0, 7.5))
    nodes.append(node2)
    node3 = Vertex(3, (0.0, 2.0))
    nodes.append(node3)
    node4 = Vertex(4, (4.0, 0.0))
    nodes.append(node4)
    node5 = Vertex(5, (4.0, 4.0))
    nodes.append(node5)
    node6 = Vertex(6, (4.0, 8.0))
    nodes.append(node6)
    node7 = Vertex(7, (8.0, 4.0))
    nodes.append(node7)

    edges = []
    edge1 = Edge(node1, node2, 0.1)
    edge1.block_status = 0
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)

    edge2 = Edge(node1, node3, 0.1)
    edge2.block_status = 0
    edges.append(edge2)
    node1.neighbors.append(node3.id)
    node3.neighbors.append(node1.id)
    
    edge3 = Edge(node2, node3, 0.1)
    edge3.block_status = 0
    edges.append(edge3)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    
    edge4 = Edge(node2, node5, 0.1)
    edge4.block_status = 0
    edges.append(edge4)
    node2.neighbors.append(node5.id)
    node5.neighbors.append(node2.id)
    
    edge5 = Edge(node2, node6, 0.1)
    edge5.block_status = 0
    edges.append(edge5)
    node2.neighbors.append(node6.id)
    node6.neighbors.append(node2.id)
    
    edge6 = Edge(node3, node4, 0.1)
    edge6.block_status = 0
    edges.append(edge6)
    node3.neighbors.append(node4.id)
    node4.neighbors.append(node3.id)
    
    edge7 = Edge(node3, node5, 0.1)
    edge7.block_status = 0
    edges.append(edge7)
    node3.neighbors.append(node5.id)
    node5.neighbors.append(node3.id)
    
    edge8 = Edge(node4, node5, 0.1)
    edge8.block_status = 0
    edges.append(edge8)
    node4.neighbors.append(node5.id)
    node5.neighbors.append(node4.id)
    
    edge9 = Edge(node7, node4, random.uniform(0.85, 0.99))
    edge9.block_status = 1
    edges.append(edge9)
    node7.neighbors.append(node4.id)
    node4.neighbors.append(node7.id)

    edge10 = Edge(node7, node5, random.uniform(0.9, 1.0))
    edge10.block_status = 1
    edges.append(edge10)
    node7.neighbors.append(node5.id)
    node5.neighbors.append(node7.id)
    
    edge11 = Edge(node6, node7, 0.1)
    edge11.block_status = 0
    edges.append(edge11)
    node6.neighbors.append(node7.id)
    node7.neighbors.append(node6.id)

    edge12 = Edge(node6, node5, 0.1)
    edge12.block_status = 0
    edges.append(edge12)
    node6.neighbors.append(node5.id)
    node5.neighbors.append(node6.id)
    robots = RobotData(robot_id = 1, position=(-3.0, 4.0), cur_vertex=start)
    return start, goal, nodes, edges, robots

def s_graph_unc():
    """Generate a simple graph for testing purposes."""
    start = 1
    goal = 4
    nodes = []
    node1 = Vertex(1, (0.0, 0.0))
    nodes.append(node1)
    node2 = Vertex(2, (4.0, 4.0))
    nodes.append(node2)
    node3 = Vertex(3, (4.0, 0.0))
    nodes.append(node3)
    node4 = Vertex(4, (8.0, 0.0))
    nodes.append(node4)
    

    edges = []
    edge1 = Edge(node1, node2, random.uniform(0.1, 0.3))
    edge1.block_status = 0
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)

    edge2 = Edge(node1, node3, random.uniform(0.2, 0.4))
    edge2.block_status = 0
    edges.append(edge2)
    node1.neighbors.append(node3.id)
    node3.neighbors.append(node1.id)
    
    edge3 = Edge(node2, node3, random.uniform(0.15, 0.3))
    edge3.block_status = 0
    edges.append(edge3)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    
    edge4 = Edge(node2, node4, random.uniform(0.08, 0.14))
    edge4.block_status = 0
    edges.append(edge4)
    node2.neighbors.append(node4.id)
    node4.neighbors.append(node2.id)
    
    edge5 = Edge(node3, node4, random.uniform(0.85, 0.99))
    edge5.block_status = 1
    edges.append(edge5)
    node3.neighbors.append(node4.id)
    node4.neighbors.append(node3.id)
    
    robots = RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)
    return start, goal, nodes, edges, robots


def disjoint_unc(): # edge 34 is blocked
    start = 1
    goal = 3
    # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
    nodes = []
    node1 =  Vertex(1, (0.0, 0))
    nodes.append(node1)
    node2 =  Vertex(2, (4.0, 0.0))
    nodes.append(node2)
    node3 =  Vertex(3, (8.0, 0.0))
    nodes.append(node3)
    node4 =  Vertex(4, (4.0, 4.0))
    nodes.append(node4)
    
    edges = []
    # edge 1
    edge1 =  Edge(node1, node2, 0.1)
    edge1.block_status = 0
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)
    # edge 2
    edge2 =  Edge(node3, node4, 0.15)
    edge2.block_status = 0
    edges.append(edge2)
    node3.neighbors.append(node4.id)
    node4.neighbors.append(node3.id)
    # edge 3
    edge3 =  Edge(node1, node4, 0.2) # length = 6.4
    edge3.block_status = 0
    edges.append(edge3)
    node1.neighbors.append(node4.id)
    node4.neighbors.append(node1.id)
    # edge 4
    edge4 =  Edge(node2, node3, 0.9)
    edge4.block_status = 1
    edges.append(edge4)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    robots = RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)
    return start, goal, nodes, edges, robots

def linear_graph_unc():
    start = 1
    goal = 3
    nodes = []
    node1 = Vertex(1, (0.0, 0.0))
    nodes.append(node1)
    node2 =  Vertex(2, (5.0, 0.0))
    nodes.append(node2)
    node3 =  Vertex(3, (15.0, 0.0))
    nodes.append(node3)
    
    edges = []
    edge1 =  Edge(node1, node2, 0.9)
    edge1.block_status = 1
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)
    edge2 =  Edge(node2, node3, 0.0)
    edge2.block_status = 0
    edges.append(edge2)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    robots = RobotData(robot_id = 1, position=(0.0, 0.0), cur_vertex=start)
    return start, goal, nodes, edges, robots
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

<<<<<<< HEAD
=======
def simple_graph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = Vertex(1, (-3.0, 4.0))
    nodes.append(node1)
    node2 = Vertex(2, (0.0, 5.5))
    nodes.append(node2)
    node3 = Vertex(3, (0.0, 2.0))
    nodes.append(node3)
    node4 = Vertex(4, (4.0, 0.0))
    nodes.append(node4)
    node5 = Vertex(5, (4.0, 4.0))
    nodes.append(node5)
    node6 = Vertex(6, (4.0, 8.0))
    nodes.append(node6)
    node7 = Vertex(7, (8.0, 4.0))
    nodes.append(node7)
    edges = []
    edge1 = Edge(node1, node2, random.uniform(0.0, 0.1))
    edge1.block_status = 0
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)

    edge2 = Edge(node1, node3, random.uniform(0.2, 0.4))
    edge2.block_status = 0
    edges.append(edge2)
    node1.neighbors.append(node3.id)
    node3.neighbors.append(node1.id)
    
    edge3 = Edge(node2, node3, random.uniform(0.15, 0.3))
    edge3.block_status = 0
    edges.append(edge3)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    
    edge4 = Edge(node2, node5, random.uniform(0.05, 0.14))
    edge4.block_status = 0
    edges.append(edge4)
    node2.neighbors.append(node5.id)
    node5.neighbors.append(node2.id)
    
    edge5 = Edge(node2, node6, random.uniform(0.0, 0.4))
    edge5.block_status = 0
    edges.append(edge5)
    node2.neighbors.append(node6.id)
    node6.neighbors.append(node2.id)
    
    edge6 = Edge(node3, node4, random.uniform(0.9, 1.0))
    edge6.block_status = 1
    edges.append(edge6)
    node3.neighbors.append(node4.id)
    node4.neighbors.append(node3.id)
    
    edge7 = Edge(node3, node5, random.uniform(0.1, 0.3))
    edge7.block_status = 0
    edges.append(edge7)
    node3.neighbors.append(node5.id)
    node5.neighbors.append(node3.id)
    
    edge8 = Edge(node4, node5, random.uniform(0.0, 0.2))
    edge8.block_status = 0
    edges.append(edge8)
    node4.neighbors.append(node5.id)
    node5.neighbors.append(node4.id)
    
    edge = Edge(node7, node4, random.uniform(0.2, 0.3))
    edge.block_status = 0
    edges.append(edge)
    node7.neighbors.append(node4.id)
    node4.neighbors.append(node7.id)


    edge = Edge(node6, node5, random.uniform(0.5, 0.7))
    edge.block_status = 0
    edges.append(edge)
    node6.neighbors.append(node5.id)
    node5.neighbors.append(node6.id)
    
    edge = Edge(node7, node5, random.uniform(0.9, 1.0))
    edge.block_status = 1
    edges.append(edge)
    node7.neighbors.append(node5.id)
    node5.neighbors.append(node7.id)
    
    edge = Edge(node6, node7, random.uniform(0.9, 1.0))
    edge.block_status = 1
    edges.append(edge)
    node6.neighbors.append(node7.id)
    node7.neighbors.append(node6.id)
    

    return nodes, edges

def disjoint_graph():
    """Generate a disjoint graph for testing purposes."""
    nodes = []
    node1 = Vertex(1, (4.0, 0))
    nodes.append(node1)
    node2 =  Vertex(2, (0.0, 4.0))
    nodes.append(node2)
    node3 =  Vertex(3, (4.0, 8.0))
    nodes.append(node3)
    node4 =  Vertex(4, (8.0, 4.0))
    nodes.append(node4)
    node5 =  Vertex(5, (4.0, 4.0))
    nodes.append(node5)
    edges = []
    edge1 =  Edge(node1, node2, random.uniform(0.0, 0.2))
    edge1.block_status = 0
    edges.append(edge1)
    edge2 =  Edge(node3, node4, 0.1)
    edge2.block_status = 0
    edges.append(edge2)
    edge4 =  Edge(node2, node5, random.uniform(0.0, 0.2))
    edge4.block_status = 0
    edges.append(edge4)
    edge6 =  Edge(node4, node5, 0.85)
    edge6.block_status = 1
    edges.append(edge6)
    edge7 =  Edge(node1, node4, 0.92)
    edge7.block_status = 1
    edges.append(edge7)
    edge8 =  Edge(node2, node3, random.uniform(0.0, 0.2))
    edge8.block_status = 0
    edges.append(edge8)
    return nodes, edges

def simple_disjoint_graph():
    """Generate a disjoint graph for testing purposes."""
    nodes = []
    node1 =  Vertex(1, (0.0, 0))
    nodes.append(node1)
    node2 =  Vertex(2, (0.0, 4.0))
    nodes.append(node2)
    node3 =  Vertex(3, (4.0, 4.0))
    nodes.append(node3)
    node4 =  Vertex(4, (5.0, 4.0))
    nodes.append(node4)
    
    edges = []
    edge1 =  Edge(node1, node2, random.uniform(0.0, 0.2))
    edge1.block_status = 0
    edges.append(edge1)
    node1.neighbors.append(node2.id)
    node2.neighbors.append(node1.id)
    edge2 =  Edge(node3, node4, 0.1)
    edge2.block_status = 0
    edges.append(edge2)
    node3.neighbors.append(node4.id)
    node4.neighbors.append(node3.id)
    edge7 =  Edge(node1, node4, 0.92)
    edge7.block_status = 1
    edges.append(edge7)
    node1.neighbors.append(node4.id)
    node4.neighbors.append(node1.id)
    edge8 =  Edge(node2, node3, random.uniform(0.0, 0.2))
    edge8.block_status = 0
    edges.append(edge8)
    node2.neighbors.append(node3.id)
    node3.neighbors.append(node2.id)
    return nodes, edges

>>>>>>> 4c64c77 (adding some tests)
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