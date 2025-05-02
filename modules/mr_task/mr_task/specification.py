import numpy as np


specifications = {}
specifications[1] = [
    lambda a: f"F {a}"
]
specifications[2] = specifications[1] + [
    lambda a, b: f"F {a} & F {b}",  # UO
    lambda a, b: f"F {a} | F {b}",  # UO
    lambda a, b: f"(!{a} U {b}) & (F {a})",  # O
    lambda a, b: f"(!{b} U {a}) & (F {b})",  # O
]
specifications[3] = specifications[2] + [
    lambda a, b, c: f"F {a} & F {b} & F {c}",  # UO
    lambda a, b, c: f"(!{a} U {b}) & (!{b} U {c}) & (F {a})",  # O
]
specifications[4] = specifications[3] + [
    lambda a, b, c, d: f"F {a} & F {b} & F {c} & F {d}",  # UO
    lambda a, b, c, d: f"((((!{a} & !{b}) U {c}) | ((!{a} & !{b}) U {d})) & (!{a} U {b})) & (F {a})",  # O
]


def get_random_specification(objects, seed=1024):
    np.random.seed(seed)
    if not objects or len(objects) < 1:
        raise ValueError("At least 1 objects are required to generate a specification")

    n = 4 if len(objects) > 4 else len(objects)
    spec = np.random.choice(specifications[n])
    num_objects = spec.__code__.co_argcount
    specs_objects = np.random.choice(objects, num_objects, replace=False)
    print(num_objects, specs_objects)
    return spec(*specs_objects)
