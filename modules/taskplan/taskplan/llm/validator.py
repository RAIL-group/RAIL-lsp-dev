class PDDLStateValidator:
    def __init__(self, problem_struct):
        self.objects = set()
        for objs in problem_struct.get('objects', {}).values():
            self.objects.update(objs)
        
        self.facts = set()
        for pred in problem_struct.get('init_predicates', []):
            # Normalize predicates to tuples of strings
            self.facts.add(tuple(str(x) for x in pred))

    def validate_and_apply(self, action):
        name = action.name.lower()
        args = [str(x) for x in action.args]

        # Check that all arguments are valid objects/locations in the problem
        for arg in args:
            if arg not in self.objects:
                print(f"Validation failed: Argument '{arg}' not found in objects list.")
                return False

        def has(pred_name, *pred_args):
            return (pred_name, *pred_args) in self.facts

        def add(pred_name, *pred_args):
            self.facts.add((pred_name, *pred_args))

        def remove(pred_name, *pred_args):
            self.facts.discard((pred_name, *pred_args))

        if name == 'pour-water':
            if len(args) != 3: return False
            pour_from, pour_to, loc = args
            if not (has('is-fillable', pour_to) and
                    has('is-at', pour_to, loc) and
                    has('filled-with-water', pour_from) and
                    has('is-holding', pour_from) and
                    has('rob-at', loc) and
                    not has('filled-with-water', pour_to) and
                    not has('filled-with-coffee', pour_to)):
                return False
            add('filled-with-water', pour_to)
            remove('filled-with-water', pour_from)
            remove('ban-move')
            remove('ban-find')

        elif name == 'pour-coffee':
            if len(args) != 3: return False
            pour_from, pour_to, loc = args
            if not (has('is-fillable', pour_to) and
                    has('is-at', pour_to, loc) and
                    has('filled-with-coffee', pour_from) and
                    has('is-holding', pour_from) and
                    has('rob-at', loc) and
                    not has('filled-with-water', pour_to) and
                    not has('filled-with-coffee', pour_to)):
                return False
            add('filled-with-coffee', pour_to)
            remove('filled-with-coffee', pour_from)
            remove('ban-move')
            remove('ban-find')

        elif name == 'make-coffee':
            if len(args) != 3: return False
            ingredient, receptacle, loc = args
            if not (has('hand-is-free') and
                    has('rob-at', loc) and
                    has('is-coffeemaker', receptacle) and
                    has('filled-with-water', receptacle) and
                    has('is-at', receptacle, loc) and
                    has('is-coffeeingredient', ingredient) and
                    has('is-at', ingredient, loc)):
                return False
            add('filled-with-coffee', receptacle)
            remove('filled-with-water', receptacle)
            remove('ban-move')
            remove('ban-find')

        elif name == 'boil':
            if len(args) != 3: return False
            boilitem, boiler, loc = args
            if not (has('hand-is-free') and
                    has('is-boilable', boilitem) and
                    has('is-at', boilitem, loc) and
                    has('is-boiler', boiler) and
                    has('is-at', boiler, loc) and
                    has('rob-at', loc) and
                    not has('is-boiled', boilitem)):
                return False
            add('is-boiled', boilitem)
            remove('ban-move')
            remove('ban-find')

        elif name == 'peel':
            if len(args) != 3: return False
            peelitem, peeler, loc = args
            if not (has('is-peelable', peelitem) and
                    has('is-at', peelitem, loc) and
                    has('is-holding', peeler) and
                    has('is-peeler', peeler) and
                    has('rob-at', loc) and
                    not has('is-peeled', peelitem)):
                return False
            add('is-peeled', peelitem)
            remove('ban-move')
            remove('ban-find')

        elif name == 'toast':
            if len(args) != 3: return False
            toastitem, toaster, loc = args
            if not (has('hand-is-free') and
                    has('is-toastable', toastitem) and
                    has('is-at', toastitem, loc) and
                    has('is-toaster', toaster) and
                    has('is-at', toaster, loc) and
                    has('rob-at', loc) and
                    not has('is-toasted', toastitem)):
                return False
            add('is-toasted', toastitem)
            remove('ban-move')
            remove('ban-find')

        elif name == 'pick':
            if len(args) != 2: return False
            obj, loc = args
            if not (has('is-pickable', obj) and
                    has('is-located', obj) and
                    has('is-at', obj, loc) and
                    has('rob-at', loc) and
                    has('hand-is-free')):
                return False
            remove('is-at', obj, loc)
            add('is-holding', obj)
            remove('hand-is-free')
            remove('ban-move')
            remove('ban-find')

        elif name == 'place':
            if len(args) != 2: return False
            obj, loc = args
            if not (not has('hand-is-free') and
                    has('rob-at', loc) and
                    has('is-holding', obj)):
                return False
            add('is-at', obj, loc)
            remove('is-holding', obj)
            add('hand-is-free')
            remove('ban-move')
            remove('ban-find')

        elif name == 'move':
            if len(args) != 2: return False
            start, end = args
            if start == end:
                return False
            if not (not has('restrict-move-to', end) and
                    not has('ban-move') and
                    has('rob-at', start)):
                return False
            remove('rob-at', start)
            add('rob-at', end)
            add('ban-move')
            add('ban-find')

        elif name == 'find':
            if len(args) != 2: return False
            obj, loc = args
            cond = (has('rob-at', loc) and
                    not has('is-located', obj) and
                    has('is-pickable', obj) and
                    has('hand-is-free'))
            if not cond:
                print(f"Validation failed for find({obj}, {loc}): "
                      f"rob-at={has('rob-at', loc)}, "
                      f"not-located={not has('is-located', obj)}, "
                      f"pickable={has('is-pickable', obj)}, "
                      f"hand-free={has('hand-is-free')}")
                return False
            add('is-located', obj)
            remove('hand-is-free')
            add('is-holding', obj)
            remove('ban-move')

        else:
            print(f"Validation failed: Unknown action name '{name}'.")
            return False

        return True
