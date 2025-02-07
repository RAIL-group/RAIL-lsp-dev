import mr_task
import numpy as np


# LOCATIONS = {"kitchen": 1, "desk": 2}
# OBJECTS = ["Knife", "Notebook", "Pen"]
# likelihoods = {
#     "kitchen": {"Knife": 1.0, "Notebook": 0.0, "Pen": 0.0},
#     "desk": {"Knife": 0.0, "Notebook": 1.0, "Pen": 1.0}
# }

GRID_SIZE = 500
LOCATIONS = {"kitchen": 1, "desk": 2, "shelf": 3, "bedroom": 4, "bathroom": 5, "storage": 6}
OBJECTS = ["Knife", "Notebook", "Pen", "Lamp", "Clock", "Toolset", "Pillow"]

# likelihoods = {
#     "kitchen": {"Knife": 0.8, "Notebook": 0.1, "Pen": 0.05, "Lamp": 0.02, "Clock": 0.1, "Toolset": 0.05, "Pillow": 0.01},
#     "desk": {"Knife": 0.1, "Notebook": 0.7, "Pen": 0.8, "Lamp": 0.3, "Clock": 0.2, "Toolset": 0.1, "Pillow": 0.05},
#     "shelf": {"Knife": 0.05, "Notebook": 0.1, "Pen": 0.05, "Lamp": 0.5, "Clock": 0.3, "Toolset": 0.2, "Pillow": 0.02},
#     "bedroom": {"Knife": 0.02, "Notebook": 0.05, "Pen": 0.05, "Lamp": 0.4, "Clock": 0.3, "Toolset": 0.05, "Pillow": 0.85},
#     "bathroom": {"Knife": 0.01, "Notebook": 0.02, "Pen": 0.02, "Lamp": 0.05, "Clock": 0.05, "Toolset": 0.02, "Pillow": 0.03},
#     "storage": {"Knife": 0.02, "Notebook": 0.03, "Pen": 0.03, "Lamp": 0.03, "Clock": 0.05, "Toolset": 0.58, "Pillow": 0.04}
# }

likelihoods = {
    "kitchen": {"Knife": 1.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "desk": {"Knife": 0.0, "Notebook": 1.0, "Pen": 1.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "shelf": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 1.0, "Toolset": 0.0, "Pillow": 0.0},
    "bedroom": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 1.0, "Clock": 1.0, "Toolset": 0.0, "Pillow": 1.0},
    "bathroom": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "storage": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 1.0, "Pillow": 0.0}
}


class ToyMap():
    def __init__(self, seed=1024):
        np.random.seed(seed)
        self.coords_locations, self.location_objects = self.generate_env()
        self.objects_in_environment = []
        for objs in self.location_objects.values():
            for obj in objs:
                if obj not in self.objects_in_environment:
                    self.objects_in_environment.append(obj)

    def generate_env(self):
        location_coords = {}
        location_objects = {loc: [] for loc in LOCATIONS}
        for loc in LOCATIONS:
            x, y = np.random.randint(0, GRID_SIZE, 2)
            location_coords[(x, y)] = loc
            # randomly sample objects in those locations
            for object in OBJECTS:
                ps = likelihoods[loc][object]
                if np.random.rand() < ps:
                    location_objects[loc].append(object)
            # object = np.random.choice(list(likelihoods[loc].keys()),
            #                           size=np.random.randint(1, 4),
            #                           p=list(likelihoods[loc].values())/np.sum(list(likelihoods[loc].values())))
            # location_objects[loc].extend(object)
        return location_coords, location_objects
