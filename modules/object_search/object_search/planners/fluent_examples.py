from fluent import Fluent
from assipUtils import State
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
            #  33: {'id': 'pillow|surface|6|14', 'name': 'pillow', 'position': (229, 25), 'type': [0, 0, 0, 1]}
    }

predictor = {'bed|4|0|0':{'plate|surface|8|15':0, 'mug|surface|8|17':0, 'pencil|surface|8|23':0, 'bowl|surface|9|24':0, 'dish|surface|9|27':0, 'pillow|surface|6|14':1},
            'sink|5|1|0':{'plate|surface|8|15':0, 'mug|surface|8|17':0, 'pencil|surface|8|23':0, 'bowl|surface|9|24':0, 'dish|surface|9|27':1, 'pillow|surface|6|14':0},
            'fridge|8|2': {'plate|surface|8|15':1, 'mug|surface|8|17':0, 'pencil|surface|8|23':0, 'bowl|surface|9|24':1, 'dish|surface|9|27':0, 'pillow|surface|6|14':1},
            'countertop|8|0':{'plate|surface|8|15':0, 'mug|surface|8|17':1, 'pencil|surface|8|23':1, 'bowl|surface|9|24':0, 'dish|surface|9|27':0, 'pillow|surface|6|14':0}
             }

robot_pos = (31, 183)
locations = ["countertop|8|0", "sink|5|1|0", "bed|4|0|0", "fridge|8|2"]
objects = ["plate|surface|8|15", "bowl|surface|9|24", "pencil|surface|8|23", 'dish|surface|9|27',  'mug|surface|8|17','pillow|surface|6|14']

#Inital Fluents, will be defined elsewhere
fluents = [Fluent("at pillow|surface|6|14 bed|4|0|0"), 
           Fluent("free robot"), Fluent("at robot countertop|8|0"),
           Fluent("near robot countertop|8|0"), ~Fluent("near robot sink|5|1|0"),
           ~Fluent("near robot bed|4|0|0"), ~Fluent("near robot fridge|8|2"),
           ~Fluent("holding robot plate|surface|8|15"), ~Fluent("holding robot bowl|surface|9|24"),
           ~Fluent("holding robot pillow|surface|6|14"), ~Fluent("holding robot pencil|surface|8|23"),
           ~Fluent("holding robot dish|surface|9|27"), ~Fluent("holding robot mug|surface|8|17")]

#Inital visited is empty
holdingObjects = set()

#Inital state
print('___Inital state___')
state = State(robot_pos, graph, holdingObjects, fluents, predictor,True)
print(state.fluents)

print('___Robot drives near the sink___')
robot_pos = (110, 85)
new_state = State(robot_pos, graph, holdingObjects, fluents, predictor, False)
print(new_state.fluents)

print('___Robot grabs the dish___')
holdingObjects.add('dish|surface|9|27')
new_state = State(robot_pos, graph, holdingObjects, fluents, predictor,False)
print(new_state.fluents)

print("___Robot is not moving___")
new_state = State(robot_pos, graph, holdingObjects, fluents,predictor , True)
print(new_state.fluents)
#delay the robot/send a signal?

print('___Robot drives away from sink toward bed___')
robot_pos = (220, 20)
new_state = State(robot_pos, graph, holdingObjects, fluents,predictor, False)
print(new_state.fluents)

print('___Robot finds the pillow on the bed___')
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
new_state = State(robot_pos, graph, holdingObjects, fluents, predictor, False)
print(new_state.fluents)