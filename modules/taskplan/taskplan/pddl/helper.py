import random
from pddlstream.algorithms.search import solve_from_pddl

import taskplan
from taskplan.planners.planner import LearnedPlanner


def generate_pddl_problem_from_struct(struct):
    '''struck has keys: 'domain_name', 'problem_name', 'objects', 
    'init_predicates', 'init_fluents', 'goal_states', 'metric'
    init_predicates is a list of strings but init fluents is a dictionary
    '''
    # Start the problem definition
    problem_str = f"(define (problem {struct['problem_name']})\n"
    problem_str += f"    (:domain {struct['domain_name']})\n"

    # Define objects
    problem_str += "    (:objects\n"
    for obj_type, obj_names in struct['objects'].items():
        problem_str += "        " + " ".join(obj_names) + " - " + obj_type + "\n"
    problem_str += "    )\n"

    # Define states
    # Define initial predicates first
    problem_str += "    (:init\n"
    for predicate in struct['init_predicates']:
        if predicate[0] == 'not':
            str_predicate = 'not (' + ' '.join(predicate[1:]) + ')'
        elif predicate[0] == 'obj-type':
            str_predicate = f'obj-type-{predicate[1]} {predicate[2]}'
        else:
            str_predicate = ' '.join(predicate)
        problem_str += f"        ({str_predicate})\n"
    # Define initial fluents next
    for fluent, values in struct['init_fluents'].items():
        str_fluent = ' '.join(fluent)
        problem_str += f"        (= ({str_fluent}) {values})\n"
    problem_str += "    )\n"

    # Define goal state
    problem_str += "    (:goal\n"
    problem_str += "        (and\n"
    for state in struct['goal_states']:
        problem_str += "            " + state + "\n"
    problem_str += "        )\n"
    problem_str += "    )\n"

    # Define metric
    if 'metric' in struct:
        problem_str += f"    (:metric {struct['metric']})\n"
    else:
        problem_str += "    (:metric minimize (total-cost))\n"

    # Close the problem definition
    problem_str += ")\n"

    return problem_str


def get_pddl_instance(whole_graph, map_data, args, learned_data=None):
    # Initialize the environment setting which containers are undiscovered
    if args.cost_type == 'known':
        init_subgoals_idx = []
    else:
        init_subgoals_idx = taskplan.utilities.utils.initialize_environment(
            whole_graph['cnt_node_idx'], args.current_seed)
    subgoal_IDs = taskplan.utilities.utils.get_container_ID(
        whole_graph['nodes'], init_subgoals_idx)
    if learned_data:
        learned_data['subgoals'] = init_subgoals_idx

    # initialize pddl related contents
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain(whole_graph)
    pddl['problem_struct'], pddl['goal'] = taskplan.pddl.problem.get_problem(
        map_data=map_data, unvisited=subgoal_IDs,
        seed=args.current_seed, cost_type=args.cost_type,
        goal_type=args.goal_type, learned_data=learned_data)
    pddl['planner'] = 'ff-astar2'  # 'max-astar'
    pddl['subgoals'] = init_subgoals_idx
    return pddl


def get_expected_cost_of_finding(partial_map, subgoals, obj_name,
                                 robot_pose, destination,
                                 learned_net, sub_pred=None):
    ''' This function calculates and returns the expected cost of finding an object
    given the partial map, initial subgoals, object name, initial robot pose, and a
    learned network path
    '''
    obj_idx = partial_map.idx_map[obj_name]
    partial_map.target_obj = obj_idx
    # avoid re-computing the subgoals predictions if already computed for obj_name
    if sub_pred is None:
        graph, subgoals = partial_map.update_graph_and_subgoals(subgoals)
        args = lambda: None
        args.network_file = learned_net
        planner = LearnedPlanner(args, partial_map, verbose=False,
                                 destination=destination)
        planner.update(graph, subgoals, robot_pose)
        sub_pred = planner.subgoals
    exp_cost, _ = (
        taskplan.core.get_best_expected_cost_and_frontier_list(
            sub_pred,
            partial_map,
            robot_pose,
            destination,
            num_frontiers_max=8,
            alternate_sampling=True))
    return round(exp_cost, 4), sub_pred


def update_problem_move(problem, end):
    x = '(rob-at '
    y = '(not (ban-move))'
    w = '(not (ban-find))'
    insert_z = None
    replaced = False
    replaced_w = False
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if x in line:
            line = '        ' + x + f'{end})'
            lines[line_idx] = line
            insert_z = line_idx + 1
        if y in line:
            line = '        (ban-move)'
            replaced = True
        if w in line:
            line = '        (ban-find)'
            replaced_w = True
    if not replaced:
        lines.insert(insert_z, '        (ban-move)')
    if not replaced_w:
        lines.insert(insert_z, '        (ban-find)')
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_pick(problem, obj, loc):
    v = '        (ban-find)'
    w = '        (ban-move)'
    x = f'        (is-holding {obj})'
    insert_x = None
    y = '        (hand-is-free)'
    z = f'        (is-at {obj} {loc})'
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif y in line:
            line = '        ' + f'(not (hand-is-free))'
            lines[line_idx] = line
        elif z in line:
            line = '        ' + f'(not (is-at {obj} {loc}))'
            lines[line_idx] = line
            insert_x = line_idx + 1
    if insert_x:
        lines.insert(insert_x, x)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_place(problem, obj, loc):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = '(not (hand-is-free))'
    y = f'(not (is-at {obj} '
    z = f'(is-holding {obj})'
    delete_z = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            line = '        ' + '(hand-is-free)'
            lines[line_idx] = line
        elif y in line:
            line = '        ' + f'(is-at {obj} {loc})'
            lines[line_idx] = line
        elif z in line:
            line = '        ' + f'(not {z})'
            delete_z = line_idx
    if delete_z:
        del lines[delete_z]
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_pourwater(problem, p_from, p_to):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'        (filled-with-water {p_from})'
    y = f'        (is-located {p_to})'
    z = f'        (filled-with-water {p_to})'
    delete_x = None
    insert_z = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            delete_x = line_idx
        elif y in line:
            insert_z = line_idx + 1
    if delete_x < insert_z:
        lines.insert(insert_z, z)
        del lines[delete_x]
    else:
        del lines[delete_x]
        lines.insert(insert_z, z)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_pourcoffee(problem, p_from, p_to):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'        (filled-with-coffee {p_from})'
    y = f'        (is-located {p_to})'
    z = f'        (filled-with-coffee {p_to})'
    delete_x = None
    insert_z = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            delete_x = line_idx
        elif y in line:
            insert_z = line_idx + 1
    if delete_x < insert_z:
        lines.insert(insert_z, z)
        del lines[delete_x]
    else:
        del lines[delete_x]
        lines.insert(insert_z, z)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_makecoffee(problem, obj):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'        (filled-with-water {obj})'
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            line = '        ' + f'(filled-with-coffee {obj})'
            lines[line_idx] = line
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_boil(problem, obj):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'(is-boilable {obj})'
    y = f'(is-boiled {obj})'
    insert_y = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            insert_y = line_idx + 1
    if insert_y:
        lines.insert(insert_y, y)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_peel(problem, obj):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'(is-peelable {obj})'
    y = f'(is-peeled {obj})'
    insert_y = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            insert_y = line_idx + 1
    if insert_y:
        lines.insert(insert_y, y)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_toast(problem, obj):
    v = '        (ban-find)'
    w = '(ban-move)'
    x = f'(is-toastable {obj})'
    y = f'(is-toasted {obj})'
    insert_y = None
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        (not (ban-find))'
            lines[line_idx] = line
        elif w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line
        elif x in line:
            insert_y = line_idx + 1
    if insert_y:
        lines.insert(insert_y, y)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def update_problem_find(problem, objs, loc):
    v = '(rob-at '
    w = '        (ban-move)'
    # first just update v and w in the lines
    lines = problem.splitlines()
    for line_idx, line in enumerate(lines):
        if v in line:
            line = '        ' + v + f'{loc})'
            lines[line_idx] = line
        if w in line:
            line = '        (not (ban-move))'
            lines[line_idx] = line

    # then update the objects
    for obj in objs:
        y = f'(not (is-located {obj}))'
        z = f'        (is-at {obj} {loc})'
        insert_z = None
        for line_idx, line in enumerate(lines):
            if y in line:
                line = f'        (is-located {obj})'
                lines[line_idx] = line
                insert_z = line_idx + 1
        if insert_z:
            lines.insert(insert_z, z)
    updated_pddl_problem = '\n'.join(lines)
    return updated_pddl_problem


def get_goals_for_one(seed, cnt_of_interest, obj_of_interest):
    random.seed(seed)
    goal_cnt = random.sample(cnt_of_interest, 1)
    goal_obj = random.sample(obj_of_interest, 1)
    task1 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)
    task = task1
    # goal_cnt = random.sample(cnt_of_interest, 1)
    # goal_obj = random.sample(obj_of_interest, 1)
    # task2 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)

    # goal_cnt = random.sample(cnt_of_interest, 1)
    # goal_obj = random.sample(obj_of_interest, 1)
    # task3 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)
    # task = [task3, task2, task1]
    # task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_two(seed, cnt_of_interest, obj_of_interest):
    random.seed(seed)
    goal_cnt = random.sample(cnt_of_interest, 2)
    goal_obj = random.sample(obj_of_interest, 2)
    task1 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)
    task = task1
    # goal_cnt = random.sample(cnt_of_interest, 2)
    # goal_obj = random.sample(obj_of_interest, 2)
    # task2 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)

    # goal_cnt = random.sample(cnt_of_interest, 2)
    # goal_obj = random.sample(obj_of_interest, 2)
    # task3 = taskplan.pddl.task.place_two_objects(goal_cnt, goal_obj)
    # task = [task3, task2, task1]
    # task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_three(seed, cnt_of_interest, obj_of_interest):
    if len(cnt_of_interest) < 3 or len(obj_of_interest) < 3:
        return None
    random.seed(seed)
    goal_cnt = random.sample(cnt_of_interest, 3)
    goal_obj = random.sample(obj_of_interest, 3)
    task1 = taskplan.pddl.task.place_three_objects(goal_cnt, goal_obj)
    task = task1
    # goal_cnt = random.sample(cnt_of_interest, 3)
    # goal_obj = random.sample(obj_of_interest, 3)
    # task2 = taskplan.pddl.task.place_three_objects(goal_cnt, goal_obj)

    # goal_cnt = random.sample(cnt_of_interest, 3)
    # goal_obj = random.sample(obj_of_interest, 3)
    # task3 = taskplan.pddl.task.place_three_objects(goal_cnt, goal_obj)
    # task = [task3, task2, task1]
    # task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_breakfast(seed, cnt_of_interest, objects):
    random.seed(seed)
    object_relations = {
        'bowl': ['egg'],
        'plate': ['apple', 'bread', 'tomato', 'potato']
    }
    pairs = []
    for object in object_relations:
        if object in objects:
            choice1 = random.choice(objects[object])
            for object2 in object_relations[object]:
                if object2 in objects:
                    choice2 = random.choice(objects[object2])
                    pair = (choice1, choice2)
                    pairs.append(pair)

    preferred_containers = ['diningtable', 'chair', 'sofa', 'bed', 'countertop']
    compatible_containers = [cnt for cnt in cnt_of_interest
                             if cnt.split('|')[0] in preferred_containers]

    if compatible_containers == [] or len(pairs) == 0:
        return None

    goal_cnt = random.choice(compatible_containers)

    task = []
    for pair in pairs:
        goal = taskplan.pddl.task.get_related_goal(goal_cnt, pair)
        task.append(goal)

    task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_coffee(seed, cnt_of_interest, objects):
    random.seed(seed)
    receptacles = ['mug', 'cup']
    compatible_receptacles = []
    for object in receptacles:
        if object in objects:
            compatible_receptacles.append(object)
            break

    preferred_containers = ['diningtable', 'chair', 'sofa', 'bed', 'countertop']
    compatible_containers = [cnt for cnt in cnt_of_interest
                             if cnt.split('|')[0] in preferred_containers]

    if compatible_containers == [] or len(compatible_receptacles) == 0:
        return None

    goal_cnt = random.choice(compatible_containers)
    task = []

    for receptacle in compatible_receptacles:
        goal = taskplan.pddl.task.get_coffee_task(goal_cnt, receptacle)
        task.append(goal)
    task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_breakfast_coffee(seed, cnt_of_interest, objects):
    random.seed(seed)
    # breakfast part
    object_relations = {
        'bowl': ['egg'],
        'plate': ['apple', 'bread', 'tomato', 'potato']
    }
    pairs = []
    for object in object_relations:
        if object in objects:
            choice1 = random.choice(objects[object])
            for object2 in object_relations[object]:
                if object2 in objects:
                    choice2 = random.choice(objects[object2])
                    pair = (choice1, choice2)
                    pairs.append(pair)

    # coffee part
    receptacles = ['mug', 'cup']
    compatible_receptacles = []
    for object in receptacles:
        if object in objects:
            compatible_receptacles.append(object)
            break

    preferred_containers = ['diningtable', 'chair', 'sofa', 'bed', 'countertop']
    compatible_containers = [cnt for cnt in cnt_of_interest
                             if cnt.split('|')[0] in preferred_containers]

    if compatible_containers == [] or len(pairs) == 0 or len(compatible_receptacles) == 0:
        return None

    goal_cnt = random.choice(compatible_containers)
    task = []
    # add a coffee-part to all breakfast pairs
    for receptacle in compatible_receptacles:
        coffee_goal = taskplan.pddl.task.get_coffee_task(
            goal_cnt, receptacle, combine=False)
        for pair in pairs:
            breakfast_goal = taskplan.pddl.task.get_related_goal(
                goal_cnt, pair, combine=False)
            task.append(f'(and {breakfast_goal} {coffee_goal})')

    task = taskplan.pddl.task.multiple_goal(task)
    return task


def get_goals_for_any_of_three(seed, cnt_of_interest, obj_of_interest):
    gen_names = []
    for obj in obj_of_interest:
        g_name = obj.split('|')[0]
        if g_name not in gen_names:
            gen_names.append(g_name)
    if len(gen_names) < 3 or len(cnt_of_interest) < 3:
        return None

    chosen = {}
    random.seed(seed)
    while len(chosen) < 3:
        goal_obj = random.sample(obj_of_interest, 1)
        g_name = goal_obj[0].split('|')[0]
        if g_name not in chosen:
            chosen[g_name] = goal_obj
    chosen = list(chosen.values())
    goal_cnt = ['initial_robot_pose']
    goal_obj = chosen[0]
    task1 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)

    goal_obj = chosen[1]
    task2 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)

    goal_obj = chosen[2]
    task3 = taskplan.pddl.task.place_one_object(goal_cnt, goal_obj)
    task = [task3, task2, task1]
    task = taskplan.pddl.task.multiple_goal(task)
    return task


def goal_provider(seed, cnt_of_interest, obj_of_interest, objects, goal_type):
    if goal_type == '1object':
        task = get_goals_for_one(seed, cnt_of_interest, obj_of_interest)
    elif goal_type == '2object':
        task = get_goals_for_two(seed, cnt_of_interest, obj_of_interest)
    elif goal_type == '3object':
        task = get_goals_for_three(seed, cnt_of_interest, obj_of_interest)
    elif goal_type == 'breakfast':
        task = get_goals_for_breakfast(seed, cnt_of_interest, objects)
    elif goal_type == 'coffee':
        task = get_goals_for_coffee(seed, cnt_of_interest, objects)
    elif goal_type == 'breakfast_coffee':
        task = get_goals_for_breakfast_coffee(seed, cnt_of_interest, objects)
    elif goal_type == 'any3':
        task = get_goals_for_any_of_three(seed, cnt_of_interest, obj_of_interest)

    return task
