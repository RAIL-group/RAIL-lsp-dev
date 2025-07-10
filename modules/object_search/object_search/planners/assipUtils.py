
class State():
    """Position of robot and dictionary of containers mapped to what’s in them) """
    def __init__(self, robot_pose, containers):
        self.robot_pose:tuple[int] = robot_pose
        self.containers:list[Container] = containers

    def getListOfContainerLocations(self):
        """Returns a list of all container locations."""
        return {container.name : container.location for container in self.containers}
    
    def getListOfKnownObjects(self):
        """Returns a list of all objects in all containers."""
        return [obj for container in self.containers for obj in container.objects]
    
class Container():
    """Container with a list of objects in it."""
    def __init__(self, id, name, location, objects=None):
        self.id = id
        self.name = name
        self.location = location
        self.objects = objects if objects is not None else []
    
    def __repr__(self):
        return f"Container({self.name}, {self.objects})"
    
    def __str__(self):
        return f"Container: {self.name} with objects: {self.objects}"

