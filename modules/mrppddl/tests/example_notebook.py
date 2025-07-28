import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, Set, Tuple
from mrppddl.src.mrppddl.core import Fluent, State, Operator, Effect, transition
import time
#graph
graph = {0: {'id': 'Apartment|0', 'name': 'apartment', 'position': (0, 0),  'type': [1, 0, 0, 0]},
             1: {'id': 'room|4', 'name': 'bedroom', 'position': (217, 67), 'type': [0, 1, 0, 0]}, 2: {'id': 'room|5', 'name': 'bathroom', 'position': (136, 60), 'type': [0, 1, 0, 0]}, 3: {'id': 'room|6', 'name': 'bedroom', 'position': (200, 184), 'type': [0, 1, 0, 0]}, 4: {'id': 'room|7', 'name': 'bathroom', 'position': (117, 234), 'type': [0, 1, 0, 0]}, 5: {'id': 'room|8', 'name': 'kitchen', 'position': (63, 137), 'type': [0, 1, 0, 0]}, 6: {'id': 'room|9', 'name': 'livingroom', 'position': (57, 37), 'type': [0, 1, 0, 0]}, 
             7: {'id': 'bed|4|0|0', 'name': 'bed', 'position': (229, 25), 'type': [0, 0, 1, 0]}, 
             11: {'id': 'sink|5|1|0', 'name': 'sink', 'position': (112, 83), 'type': [0, 0, 1, 0]}, 
             20: {'id': 'fridge|8|2', 'name': 'fridge', 'position': (114, 144), 'type': [0, 0, 1, 0]}, 
             18: {'id': 'countertop|8|0', 'name': 'countertop', 'position': (31, 183), 'type': [0, 0, 1, 0]},
             28: {'id': 'plate|surface|8|15', 'name': 'plate', 'position': (114,144), 'type': [0, 0, 0, 1]}, 
             29: {'id': 'mug|surface|8|17', 'name': 'mug', 'position': (31,183), 'type': [0, 0, 0, 1]}, 
             30: {'id': 'pencil|surface|8|23', 'name': 'pencil', 'position': (31, 183), 'type': [0, 0, 0, 1]}, 
             31: {'id': 'bowl|surface|9|24', 'name': 'bowl', 'position': (114, 144), 'type': [0, 0, 0, 1]}, 
             32: {'id': 'dish|surface|9|27', 'name': 'dish', 'position': (112, 83), 'type': [0, 0, 0, 1]},
             33: {'id': 'pillow|surface|6|14', 'name': 'pillow', 'position': (229, 25), 'type': [0, 0, 0, 1]}
    }

robot_graph = {0: {'id': 'Apartment|0', 'name': 'apartment', 'position': (0, 0),  'type': [1, 0, 0, 0]},
             1: {'id': 'room|4', 'name': 'bedroom', 'position': (217, 67), 'type': [0, 1, 0, 0]}, 2: {'id': 'room|5', 'name': 'bathroom', 'position': (136, 60), 'type': [0, 1, 0, 0]}, 3: {'id': 'room|6', 'name': 'bedroom', 'position': (200, 184), 'type': [0, 1, 0, 0]}, 4: {'id': 'room|7', 'name': 'bathroom', 'position': (117, 234), 'type': [0, 1, 0, 0]}, 5: {'id': 'room|8', 'name': 'kitchen', 'position': (63, 137), 'type': [0, 1, 0, 0]}, 6: {'id': 'room|9', 'name': 'livingroom', 'position': (57, 37), 'type': [0, 1, 0, 0]}, 
             7: {'id': 'bed|4|0|0', 'name': 'bed', 'position': (229, 25), 'type': [0, 0, 1, 0]}, 
             11: {'id': 'sink|5|1|0', 'name': 'sink', 'position': (112, 83), 'type': [0, 0, 1, 0]}, 
             20: {'id': 'fridge|8|2', 'name': 'fridge', 'position': (114, 144), 'type': [0, 0, 1, 0]}, 
             18: {'id': 'countertop|8|0', 'name': 'countertop', 'position': (31, 183), 'type': [0, 0, 1, 0]},
    }

predictor = {
                'sink|3|1|0': {
                    'faucet|3|1|1':1

                },
                
                'armchair|2|2|0':{
                    'plate|8|15':0,
                    'mug|8|17':0
                },

                'diningtable|2|3|0': {
                    'plate|8|15':0

                },

                'bed|2|0|0':{
                    'pillow|2|0|1':1,
                    'blanket|2|0|2':1
                }
            }
            
def compute_fluents(
    robot_pos: Tuple[int,int],
    graph: Dict[int,Dict],
    holding: Set[str],
    searched: Set[str] = set()
) -> Set[Fluent]:
    fs: Set[Fluent] = set()
    # "free(robot)" – here we assume the robot is always free
    fs.add(Fluent("free", "robot"))
    
    # Track positions of all objects and their locations
    locations = {id: node for id, node in graph.items() if node["type"] == [0, 0, 1, 0]}  # rooms/locations
    
    for node in graph.values():
        obj_id = node["id"]
        x,y = node["position"]
        manhattan = abs(robot_pos[0]-x) + abs(robot_pos[1]-y)
        
        # Handle robot location and proximity
        if node["type"] == [0, 0, 1, 0]:  # location/room
            if robot_pos == (x,y):
                fs.add(Fluent("at", "robot", obj_id))
            else:
                fs.add(~Fluent("at", "robot", obj_id))
            # near(robot, obj) (≤20 units)

            if node["id"] in searched:
                fs.add(Fluent("searched", obj_id))
            else:
                print(searched)
                fs.add(~Fluent("searched", obj_id))
                
        # Handle object locations
        if node["type"] == [0, 0, 0, 1]:  # objects
            # Find which location the object is in/near
            for loc_id, loc in locations.items():
                loc_x, loc_y = loc["position"]
                obj_to_loc_dist = abs(x - loc_x) + abs(y - loc_y)
                if obj_to_loc_dist <= 20:  # Object is at/near this location
                    fs.add(Fluent("at", obj_id, loc["id"]))
                else:
                    fs.add(~Fluent("at", obj_id, loc["id"]))
        
        # Handle holding state
        if obj_id in holding:
            fs.add(Fluent("holding", "robot", obj_id))
        else:
            fs.add(~Fluent("holding", "robot", obj_id))

    return {fluent for fluent in fs if not fluent.negated}

def get_position_for_location(graph, location_id):
    """Get the (x,y) position for a given location ID from the graph"""
    for node in graph.values():
        if node['id'] == location_id:
            return node['position']
    return None

def update_robot_position(graph, robot_pos, target_loc):
    """Update robot position when moving to a target location"""
    target_pos = get_position_for_location(graph, target_loc)
    if target_pos:
        return target_pos
    return robot_pos

class RobotState(State):
    def __init__(self, time: float = 0, fluents=None, robot_pos=None, holding=None, searched_locations=None):
        super().__init__(time=time, fluents=fluents)
        self.robot_pos = robot_pos or (0, 0)
        self.holding = holding or set()
        self.searched_locations = searched_locations or set()
    
    def copy(self):
        return RobotState(
            time=self.time,
            fluents=self.fluents,
            robot_pos=self.robot_pos,
            holding=self.holding.copy(), 
            searched_locations=self.searched_locations.copy()
        )
    
def transition_with_position(state: RobotState, action, graph, robot_graph):
    if not state.satisfies_precondition(action):
        raise ValueError("Precondition not satisfied")
    
    new_state = state.copy()
    
    # Extract target location from action name for move actions
    if "move_to" in action.name or "search" in action.name:
        # Format: "move_visit robot loc_from loc_to"
        target_loc = action.name.split()[-1]
        new_state.robot_pos = update_robot_position(graph, state.robot_pos, target_loc)
        #update robot_graph
    
    # Handles search actions
    if "search" in action.name:
        print('hi')
        print(action.name.split())
        new_state.searched_locations.add(action.name.split()[2])  # Add the location being searched
        for idx in graph:
            if graph[idx]['type'] == [0, 0, 0, 1] and graph[idx]["position"] == new_state.robot_pos:
                print('yo')
                robot_graph[idx] = graph[idx]
                
    # Handle pickup actions
    elif "pickup" in action.name:
        action_parts = action.name.split()
        obj_id = action_parts[2]
        new_state.holding.add(obj_id)
        # Update object position in graph to match robot position
    
    
    for node in graph.values():
        if node['id'] in new_state.holding:
            node['position'] = new_state.robot_pos
            break
    
    # Apply standard fluent updates through core transition
    outcomes = transition(state, action)
    if outcomes:
        next_state = outcomes[0][0]
        new_state.time = next_state.time
        
        # Compute expected fluents based on current world state
        expected_fluents = compute_fluents(new_state.robot_pos, robot_graph, new_state.holding, new_state.searched_locations)
        
        # Merge effects from action with computed fluents
        action_effects = {f for f in next_state.fluents if f.name in {"visited", "waited"}}  # Keep track of special fluents
        new_state.fluents = expected_fluents | action_effects
        
        # Validate effects were properly applied
        if "pickup" in action.name:
            obj_id = action.name.split()[2]
            # print(action.name.split())
            if not any(f.name == "holding" and f.args == ("robot", obj_id) for f in new_state.fluents):
                raise ValueError(f"Pickup action failed: {obj_id} not in holding fluents")
        elif "move_to" in action.name:
            target_loc = action.name.split()[-1]
            if not any(f.name == "at" and f.args == ("robot", target_loc) for f in new_state.fluents):
                raise ValueError(f"Move action failed: robot not at {target_loc}")
    
    return new_state


# Modify move operator effects to update positions
move_time_fn = lambda *args: random.random() + 5.0  #noqa: E731
move_to_op = Operator(
    name="move_to",
    parameters=[("?robot", "robot"), ("?loc_to", "location")],
    preconditions=[Fluent("free ?robot"), ~Fluent("at robot ?loc_to")],
    effects=[
        Effect(0, [], {Fluent("not free ?robot")}),  # Just remove free, location change handled by compute_fluents
        Effect((move_time_fn, ["?robot", "?loc_to"]), [], 
               {Fluent("free ?robot"), Fluent("at ?robot ?loc_to")})
    ])

wait_op = Operator(
    name="wait",
    parameters=[("?robot", "robot")],
    preconditions=[Fluent("free ?robot"), Fluent("not waited ?robot")],
    effects=[Effect(time=0, resulting_fluents={Fluent("not free ?robot")}),
             Effect(time=10, resulting_fluents={Fluent("free ?robot"), Fluent("waited ?robot")})])

pickup_op = Operator(
    name="pickup",
    parameters=[("?robot", "robot"), ("?obj", "object"), ("?loc", "location")],
    preconditions=[Fluent("at ?robot ?loc"), Fluent("at ?obj ?loc"), Fluent("free ?robot"), Fluent("not holding ?robot ?obj")],
    effects=[Effect(time=0, resulting_fluents={Fluent("not free ?robot"), Fluent("not at ?robot ?obj")}),
             Effect(time=5, resulting_fluents={Fluent("holding ?robot ?obj")})])

search_op = Operator(
    name="search",
    parameters=[("?robot", "robot"), ("?loc_to", "location")],
    preconditions=[
        Fluent("free ?robot"), 
        ~Fluent("searched ?loc_to"), 
        # ~Fluent("found ?object"),
        # Fluent("chosen ?object")  # Only allow searching for the chosen object
    ],
    effects=[
        Effect(
            time=0,
            resulting_fluents={
                ~Fluent("free ?robot"),
                Fluent("searched ?loc_to ?object")
            }
        ),
        Effect(
            time=(move_time_fn, ["?robot", "?loc_to"]),
            resulting_fluents={Fluent("at ?robot ?loc_to"), Fluent("searched ?loc_to")},
        )
    ])


objects_by_type = {
    "robot": ["robot"],
    "location": [node["id"] for node in graph.values() if node["type"] == [0, 0, 1, 0]],
    "object": [node["id"] for node in graph.values() if node["type"] == [0, 0, 0, 1]]


}
all_actions = (
    move_to_op.instantiate(objects_by_type)
    + wait_op.instantiate(objects_by_type)
    + pickup_op.instantiate(objects_by_type)
    + search_op.instantiate(objects_by_type)
)
# print("All actions:", all_actions)

# ——————————————————————————————————————————————————————————————————————————————
# Simulation steps
holding_objects = set()
searched_locations = set()

# desired_action = input("Choose from the following actions: ")
print("\n")
print("Available locations to start from:")
for loc_id in objects_by_type["location"]:
    print(f"- {loc_id}")

start_loc = input("Choosse a starting location for the robot: ")
print("\n")
# Validate and set robot starting position
start_pos = get_position_for_location(graph, start_loc)
if start_pos is None:
    print(f"Invalid location: {start_loc}. Defaulting to (0, 0).")
    robot_pos = (0, 0)
else:
    robot_pos = start_pos

    
fluents   = compute_fluents(robot_pos, robot_graph, holding_objects, searched_locations)
state     = RobotState(time=0.0, fluents=fluents, robot_pos=robot_pos, holding=holding_objects)
print("___Initial state___")
print(f"Robot position: {state.robot_pos}")
print(state.fluents)
print(robot_graph)
desired_action = ''
while desired_action != "exit":
    print ("Available actions from current state")
    available_actions = [a for a in all_actions if state.satisfies_precondition(a)]

    if not available_actions:
        print ("no available action from current state selected")
        sys.exit()
    for i, action in enumerate(available_actions):
        print(f"{i+1}. {action.name}")
    print("\nType 'exit' to quit.")
    print("Enter the number of the action or type the full action name")
    desired_action = input("Enter choice: ")

    matched = False
    # Try index-based selection first
    if desired_action.isdigit():
        idx = int(desired_action) - 1
        if 0 <= idx < len(available_actions):
            action = available_actions[idx]
            state = transition_with_position(state, action, graph, robot_graph)
            matched = True
            print("\n_________________________________________________________________________________\n")
            print(state)
            matched = True
    else:
        # Fall back to name-based selection
        for action in available_actions:
            if desired_action == str(action.name):
                state = transition_with_position(state, action, graph, robot_graph)
                matched = True
                print("\n_________________________________________________________________________________\n")
                print(state)
                matched = True
                break

    if not matched and desired_action.lower() != "exit":
        print ("Invalid action. Make sure it is exactly copied.")




