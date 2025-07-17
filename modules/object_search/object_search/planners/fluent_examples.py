from fluent import Fluent
from assipUtils import State

#graph
nodes = {0: {'id': 'Apartment|0', 'name': 'apartment', 'position': (0, 0), 'type': [1, 0, 0, 0]}, 1: {'id': 'room|4', 'name': 'bedroom', 'position': (217, 67), 'type': [0, 1, 0, 0]}, 2: {'id': 'room|5', 'name': 'bathroom', 'position': (136, 60), 'type': [0, 1, 0, 0]}, 3: {'id': 'room|6', 'name': 'bedroom', 'position': (200, 184), 'type': [0, 1, 0, 0]}, 4: {'id': 'room|7', 'name': 'bathroom', 'position': (117, 234), 'type': [0, 1, 0, 0]}, 5: {'id': 'room|8', 'name': 'kitchen', 'position': (63, 137), 'type': [0, 1, 0, 0]}, 6: {'id': 'room|9', 'name': 'livingroom', 'position': (57, 37), 'type': [0, 1, 0, 0]}, 7: {'id': 'bed|4|0|0', 'name': 'bed', 'position': (229, 25), 'type': [0, 0, 1, 0]}, 8: {'id': 'dresser|4|2', 'name': 'dresser', 'position': (182, 95), 'type': [0, 0, 1, 0]}, 9: {'id': 'safe|4|3', 'name': 'safe', 'position': (229, 6), 'type': [0, 0, 1, 0]}, 10: {'id': 'toilet|5|0', 'name': 'toilet', 'position': (110, 15), 'type': [0, 0, 1, 0]}, 11: {'id': 'sink|5|1|0', 'name': 'sink', 'position': (112, 83), 'type': [0, 0, 1, 0]}, 12: {'id': 'garbagecan|5|3', 'name': 'garbagecan', 'position': (152, 11), 'type': [0, 0, 1, 0]}, 13: {'id': 'bed|6|0|0', 'name': 'bed', 'position': (246, 202), 'type': [0, 0, 1, 0]}, 14: {'id': 'dresser|6|1', 'name': 'dresser', 'position': (254, 181), 'type': [0, 0, 1, 0]}, 15: {'id': 'toilet|7|0', 'name': 'toilet', 'position': (115, 251), 'type': [0, 0, 1, 0]}, 16: {'id': 'sink|7|1|0', 'name': 'sink', 'position': (126, 248), 'type': [0, 0, 1, 0]}, 17: {'id': 'garbagecan|7|3', 'name': 'garbagecan', 'position': (113, 210), 'type': [0, 0, 1, 0]}, 18: {'id': 'countertop|8|0', 'name': 'countertop', 'position': (31, 183), 'type': [0, 0, 1, 0]}, 19: {'id': 'garbagebag|8|1', 'name': 'garbagebag', 'position': (39, 153), 'type': [0, 0, 1, 0]}, 20: {'id': 'fridge|8|2', 'name': 'fridge', 'position': (114, 144), 'type': [0, 0, 1, 0]}, 21: {'id': 'diningtable|8|3|0', 'name': 'diningtable', 'position': (5, 107), 'type': [0, 0, 1, 0]}, 22: {'id': 'chair|8|3|1', 'name': 'chair', 'position': (28, 107), 'type': [0, 0, 1, 0]}, 23: {'id': 'chair|8|3|2', 'name': 'chair', 'position': (5, 107), 'type': [0, 0, 1, 0]}, 24: {'id': 'garbagecan|8|4', 'name': 'garbagecan', 'position': (81, 186), 'type': [0, 0, 1, 0]}, 25: {'id': 'tvstand|9|0|0', 'name': 'tvstand', 'position': (83, 17), 'type': [0, 0, 1, 0]}, 26: {'id': 'diningtable|9|1', 'name': 'diningtable', 'position': (77, 41), 'type': [0, 0, 1, 0]}, 27: {'id': 'sofa|9|2', 'name': 'sofa', 'position': (57, 21), 'type': [0, 0, 1, 0]}, 28: {'id': 'plate|surface|8|15', 'name': 'plate', 'position': (5, 107), 'type': [0, 0, 0, 1]}, 29: {'id': 'mug|surface|8|17', 'name': 'mug', 'position': (5, 107), 'type': [0, 0, 0, 1]}, 30: {'id': 'pencil|surface|8|23', 'name': 'pencil', 'position': (5, 107), 'type': [0, 0, 0, 1]}, 31: {'id': 'bowl|surface|9|24', 'name': 'bowl', 'position': (77, 41), 'type': [0, 0, 0, 1]}, 32: {'id': 'plate|surface|9|27', 'name': 'plate', 'position': (77, 41), 'type': [0, 0, 0, 1]}}
ourNodes = {0: {'id': 'Apartment|0', 'name': 'apartment', 'position': (0, 0),  'type': [1, 0, 0, 0]},
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
             33: {'id': 'pillow|surface|6|14', 'name': 'pillow', 'position': (229, 25), 'type': [0, 0, 0, 1]}}

robot_pos = (31, 183)
locations = ["countertop|8|0", "sink|5|1|0", "bed|4|0|0", "fridge|8|2"]
objects = ["plate|surface|8|15", "bowl|surface|9|24", "pencil|surface|8|23", 'dish|surface|9|27',  'mug|surface|8|17','pillow|surface|6|14']
#Inital Fluents, will be defined elsewhere
#add fluent for testing "at" Fluent("at robot countertop|8|0") --> should be true at the start
fluents = [Fluent("at robot countertop|8|0"), Fluent("near robot countertop|8|0"), ~Fluent("near robot sink|5|1|0"),
           ~Fluent("near robot bed|4|0|0"), ~Fluent("near robot fridge|8|2"),
           ~Fluent("holding robot plate|surface|8|15"), ~Fluent("holding robot bowl|surface|9|24"),
           ~Fluent("holding robot pillow|surface|6|14"), ~Fluent("holding robot pencil|surface|8|23"),
           ~Fluent("holding robot dish|surface|9|27"), ~Fluent("holding robot mug|surface|8|17")]
#Inital visited
holdingObjects = set()
#Inital state
print('___Inital state___')
state = State(robot_pos, ourNodes, holdingObjects, fluents)
print(state.fluents)
#Robot now drives near the sink
print('___Robot drives near the sink___')
robot_pos = (110, 85)
new_state = State(robot_pos, ourNodes, holdingObjects, fluents)
print(new_state.fluents)
#Grabs the plate
print('___Robot grabs the plate___')
holdingObjects.add('dish|surface|9|27')
new_state = State(robot_pos, ourNodes, holdingObjects, fluents)
print(new_state.fluents)
#Robot drives away from sink toward bed
print('___Robot drives away from sink toward bed___')
robot_pos = (220, 20)
new_state = State(robot_pos, ourNodes, holdingObjects, fluents)
print(new_state.fluents)