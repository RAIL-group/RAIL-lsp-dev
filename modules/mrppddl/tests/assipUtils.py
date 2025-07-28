import math
from src.mrppddl.core import Fluent

# from modules.procthor.procthor.scenegraph import SceneGraph
     
class Condition():    
    def __init__(self, objectOne, objectTwo, robot_pose, graph, holding, visited, probs, target_objs, isFree=False):
        self.graph = graph
        self.free = isFree
        self.objectOne = self.get_node_info(objectOne) if objectOne != 'robot' else robot_pose
        self.objectTwo = self.get_node_info(objectTwo) if objectTwo != None else ({}, 0)
        self.robot_pose = robot_pose
        self.holding = holding
        self.visited = visited
        self.probs = probs
        self.target_objs = target_objs

    def check(self, fType):
        if fType == 'near':
            return self.isNear()
        elif fType == 'holding':
            
            return self.isHolding()
        elif fType == 'at':
            return self.isAt()
        elif fType == 'free':
            return self.isFree()
        
    def isNear(self):
        # print("this is the near distance: ", str(math.sqrt((self.objectOne[0]-self.objectTwo[0]['position'][0])**2) + math.sqrt((self.objectOne[1]-self.objectTwo[0]['position'][1])**2)))
        return math.sqrt((self.objectOne[0]-self.objectTwo[0]['position'][0])**2) + math.sqrt((self.objectOne[1]-self.objectTwo[0]['position'][1])**2) <= 20
    
    def isHolding(self):
        # assuming the robot is at a location, it has picked up the target object, how we are going to add to holding
        print("object two ", str(self.get_node_info_name(self.objectTwo[0])[0]))
        print("targetobjs ", str(self.target_objs))
        #never getting past the bellow line, maybe because one is a set the other is a string
        if self.get_node_info_name(self.objectTwo[0])[0] == self.target_objs:    
            if self.objectOne[0] == self.get_node_info_name(self.objectTwo[0])[0]['position'][0] and self.objectOne[1] == self.get_node_info_name(self.objectTwo[0])[0][1]:
                self.holding.append(self.target_objs["id"])
                return True

        if isinstance(self.objectOne[0], str):
            return self.objectTwo[0] in self.holding
        elif isinstance(self.objectOne[0], dict):
            return self.objectTwo[0]['id'] in self.holding

    def isAt(self):
       
        if isinstance(self.objectOne[0], int):
            return self.objectOne[0] == self.objectTwo[0]['position'][0] and self.objectOne[1] == self.objectTwo[0]['position'][1]
        elif isinstance(self.objectOne[0], str):
            if self.objectTwo[0]['id'] in self.visited:
                return False
            return self.predictor(self.objectOne[0], self.objectTwo[0]['id']) == 1
        elif isinstance(self.objectOne[0], dict):
            return self.objectOne[0]['position'] == self.objectTwo[0]['position']

    def isFree(self):
        return self.free
    
    def get_node_info(self, node_name: str):
        for node_id in self.graph:
            info = self.graph[node_id]
            if info['id'] == node_name:
                return (info, node_id)
        return (node_name, -1)
    
    def get_node_info_name(self, node_name: str):
        node_name = node_name["name"]
        for node_id in self.graph:
            info = self.graph[node_id]
            if info['name'] == node_name:
                return (info, node_id)
        return (node_name, -1)
    
    def predictor(self, id1, id2):
        #id2 is the container, id1 is the object
        if self.probs[id2].get(id1) is not None:
            return self.probs[id2][id1]

class State():
    """Position of robot and dictionary of containers mapped to what’s in them) """
    def __init__(self, robot_pose, graph, holding, visited, fluents:list[Fluent], probs, target_objs, isFree=False):
        self.robot_pose:tuple[int] = robot_pose
        self.isFree = isFree
        self.graph = graph
        self.holding = holding
        self.visited = visited
        self.probs = probs
        self.target_objs = target_objs
        self.found_objs = []
        self.fluents:list[Fluent] = self.updateFluents(fluents)
    
        
    def updateFluents(self, fluents):
        for idx in range(len(fluents)):
            parameters = fluents[idx].args
            # print(parameters, fluent.name)
            nodeOne = parameters[0]
            fType = fluents[idx].name
            nodeTwo = parameters[1] if len(parameters) >= 2 else None

            condition = Condition(nodeOne, nodeTwo, self.robot_pose, self.graph, self.holding,self.visited,self.probs, self.target_objs, self.isFree)
           
            if condition.check(fType):
                
                    if fluents[idx].negated:
                        fluents[idx] = ~fluents[idx]
                    else:
                        pass
            # If the type is holding, it can't become negative once it is positive, assuming the robot wont drop an object
            elif not fluents[idx].negated and fType != "holding":
                fluents[idx] = ~fluents[idx]     

        return [fluent for fluent in fluents if not fluent.negated]
    
class Container():
    """Container with a list of objects in it."""
    def __init__(self, id, name, location, objects=None):
        self.id = id
        self.name = name
        self.location = location
        self.objects = objects if objects is not None else []
    
    def __repr__(self):
        return f"Container({self.name}, {self.objects})"
    

def graphToFluents(graph):
    fluents = []
    for x in graph:
        if graph[x]["type"] == [0, 0, 1, 0]:
            string = "near robot " + graph[x]["id"]
            stringTwo = "at robot " + graph[x]["id"]
            fluents.append(~Fluent(string))
            fluents.append(~Fluent(stringTwo))
    return fluents