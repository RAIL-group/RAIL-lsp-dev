import math
from fluent import Fluent
# from modules.procthor.procthor.scenegraph import SceneGraph
     
class Condition():
    def __init__(self, objectOne, objectTwo, robot_pose, graph, holding, isFree=False):
        self.graph = graph
        self.free = isFree
        self.objectOne = self.get_node_info(objectOne) if objectOne != 'robot' else robot_pose
        self.objectTwo = self.get_node_info(objectTwo) if objectTwo != None else ({}, 0)
        self.robot_pose = robot_pose
        self.holding = holding
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
        # print("This is the distance: " + str(math.sqrt((self.objectOne[0]-self.objectTwo[0]['position'][0])**2) + math.sqrt((self.objectOne[1]-self.objectTwo[0]['position'][1])**2)))
        # if self.objectTwo[0]['id'] == 'diningtable': print(self.objectTwo)
        return math.sqrt((self.objectOne[0]-self.objectTwo[0]['position'][0])**2) + math.sqrt((self.objectOne[1]-self.objectTwo[0]['position'][1])**2) <= 20
    def isHolding(self):
        # print(self.objectTwo[0]['id'], self.holding)
        return self.objectTwo[0]['id'] in self.holding
    def isAt(self):
        # print("This is the distance: " + str(math.sqrt((self.objectOne[0]-self.objectTwo[0]['position'][0])**2) + math.sqrt((self.objectOne[1]-self.objectTwo[0]['position'][1])**2)))
        # if self.objectTwo[0]['id'] == 'diningtable': print(self.objectTwo)
        return self.objectOne[0] == self.objectTwo[0]['position'][0] and self.objectOne[1]==self.objectTwo[0]['position'][1]
    def isFree(self):
        return self.free
    
    def get_node_info(self, node_name: str) -> tuple[dict, int]:
        
        for node_id in self.graph:
            info = self.graph[node_id]
            if info['id'] == node_name:
                return (info, node_id)
        return (None, -1)
       

class State():
    """Position of robot and dictionary of containers mapped to what’s in them) """
    def __init__(self, robot_pose, graph, holding, fluents:list[Fluent], isFree=False):
        self.robot_pose:tuple[int] = robot_pose
        self.isFree = isFree
        self.graph = graph
        self.holding = holding
        self.fluents:list[Fluent] = self.updateFluents(fluents)

    def updateFluents(self, fluents):
        for idx in range(len(fluents)):
            parameters = fluents[idx].args
            # print(parameters, fluent.name)
            nodeOne = parameters[0]
            fType = fluents[idx].name
            nodeTwo = parameters[1] if len(parameters) >= 2 else None
            condition = Condition(nodeOne, nodeTwo, self.robot_pose, self.graph, self.holding, self.isFree)
           
            if condition.check(fType):
                if fluents[idx].negated:
                    fluents[idx] = ~fluents[idx]
                else:
                    pass
            elif not fluents[idx].negated:
                fluents[idx] = ~fluents[idx]     

        return [fluent for fluent in fluents if not fluent.negated]

    # def getListOfContainerLocations(self):
    #     """Returns a list of all container locations."""
    #     return {container.name : container.location for container in self.containers}
    
    # def getListOfKnownObjects(self):
    #     """Returns a list of all objects in all containers."""
    #     return [obj for container in self.containers for obj in container.objects]
    
class Container():
    """Container with a list of objects in it."""
    def __init__(self, id, name, location, objects=None):
        self.id = id
        self.name = name
        self.location = location
        self.objects = objects if objects is not None else []
    
    def __repr__(self):
        return f"Container({self.name}, {self.objects})"