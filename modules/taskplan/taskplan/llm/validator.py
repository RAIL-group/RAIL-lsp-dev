import copy
import taskplan.pddl.helper

class PDDLStateValidator:
    def __init__(self, problem_struct):
        self.objects = set()
        for objs in problem_struct.get('objects', {}).values():
            self.objects.update(objs)

        self.mock_problem = copy.deepcopy(problem_struct)

    def validate_and_apply(self, action):
        name = action.name.lower()
        args = [str(x) for x in action.args]

        # Check that all arguments are valid objects/locations in the problem
        for arg in args:
            if arg not in self.objects:
                print(f"Validation failed: Argument '{arg}' not found in objects list.")
                return False

        def has(pred_name, *pred_args):
            return (pred_name, *pred_args) in self.mock_problem['init_predicates']

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
            taskplan.pddl.helper.update_problem_pourwater(self.mock_problem, pour_from, pour_to)

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
            taskplan.pddl.helper.update_problem_pourcoffee(self.mock_problem, pour_from, pour_to)

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
            taskplan.pddl.helper.update_problem_makecoffee(self.mock_problem, receptacle)

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
            taskplan.pddl.helper.update_problem_boil(self.mock_problem, boilitem)

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
            taskplan.pddl.helper.update_problem_peel(self.mock_problem, peelitem)

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
            taskplan.pddl.helper.update_problem_toast(self.mock_problem, toastitem)

        elif name == 'pick':
            if len(args) != 2: return False
            obj, loc = args
            if not (has('is-pickable', obj) and
                    has('is-located', obj) and
                    has('is-at', obj, loc) and
                    has('rob-at', loc) and
                    has('hand-is-free')):
                return False
            taskplan.pddl.helper.update_problem_pick(self.mock_problem, obj, loc)

        elif name == 'place':
            if len(args) != 2: return False
            obj, loc = args
            if not (not has('hand-is-free') and
                    has('rob-at', loc) and
                    has('is-holding', obj)):
                return False
            taskplan.pddl.helper.update_problem_place(self.mock_problem, obj, loc)

        elif name == 'move':
            if len(args) != 2: return False
            start, end = args
            if start == end:
                return False
            if not (not has('restrict-move-to', end) and
                    not has('ban-move') and
                    has('rob-at', start)):
                return False
            taskplan.pddl.helper.update_problem_move(self.mock_problem, end)

        elif name == 'find':
            if len(args) != 2: return False
            obj, loc = args
            
            if 'subgoals' in self.mock_problem and loc not in self.mock_problem['subgoals']:
                print(f"Validation failed for find({obj}, {loc}): Location '{loc}' has already been searched or is not a valid subgoal.")
                return False
                
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
            
            if ('ban-move',) in self.mock_problem['init_predicates']:
                self.mock_problem['init_predicates'].remove(('ban-move',))

        else:
            print(f"Validation failed: Unknown action name '{name}'.")
            return False

        return True
