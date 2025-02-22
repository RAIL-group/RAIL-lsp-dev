import numpy as np


specifications = {
    1: [
        lambda a: f"F {a}"
    ],
    2: [
        lambda a, b: f"F {a} & F {b}"
    ],
    3: [
        lambda a, b: f"F {a} & F {b}",
        lambda a, b, c: f"F {a} & F {b} & F {c}"
    ],
}


def get_random_specification(objects, seed=1024):
    np.random.seed(seed)
    if not objects or len(objects) < 2:
        raise ValueError("At least 2 objects are required to generate a specification")

    n = 3 if len(objects) > 3 else len(objects)
    spec = np.random.choice(specifications[n])
    num_objects = spec.__code__.co_argcount
    specs_objects = np.random.choice(objects, num_objects, replace=False)
    print(num_objects, specs_objects)
    return spec(*specs_objects)
