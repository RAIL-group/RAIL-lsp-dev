from taskplan.pddl.helper import get_expected_cost_of_finding, goal_provider
from taskplan.utilities.utils import get_robots_room_coords, get_action_costs
from procthor.utils import get_generic_name, get_cost


pre_compute = {}

grid_cost = {}


def get_problem(map_data, unvisited, seed=0, cost_type=None, goal_type='breakfast', learned_data=None):
    costs = get_action_costs()
    obj_of_interest = []
    cnt_of_interest = []
    containers = map_data.containers
    robot_room_coord = get_robots_room_coords(
        map_data.occupancy_grid, map_data.get_robot_pose(), map_data.rooms)
    objects = {
        'init_r': ['initial_robot_pose']
    }
    init_predicates = [
        ('rob-at', 'initial_robot_pose'),
        ('hand-is-free',),
    ]
    init_fluents = {
        ('total-cost',): 0,
    }
    if goal_type != 'any3':
        init_predicates.append(('restrict-move-to', 'initial_robot_pose'))
    for container in containers:
        cnt_name = container['id']
        cnt_of_interest.append(cnt_name)
        gen_name = get_generic_name(cnt_name)
        if gen_name not in objects:
            objects[gen_name] = [cnt_name]
        else:
            objects[gen_name].append(cnt_name)
        children = container.get('children')
        if children is not None:
            for child in children:
                child_name = child['id']
                obj_of_interest.append(child_name)
                gen_name_child = get_generic_name(child_name)

                if gen_name_child not in objects:
                    objects[gen_name_child] = [child_name]
                else:
                    objects[gen_name_child].append(child_name)

                cnt_names = ['initial_robot_pose']
                cnt_names += [loc['id'] for loc in containers]
                pred_sub = None
                if cnt_name in unvisited:
                    # Object is in the unknown space
                    init_predicates.append(('not', 'is-located', child_name))

                    # The expected find cost needs to be computed via the
                    # model later on. But here we use the optimistic find cost

                    # --- ROOM FOR IMPROVEMENT --- #
                    # if either of the from-loc/to-loc is in subgoals then
                    # the optimistic assumtion would be the missing object can
                    # be found in either. So, taking the distance of from-loc
                    # to to-loc is sufficient
                    for from_loc in cnt_names:
                        for to_loc in cnt_names:
                            # for the optimistic case, we add the fixed find cost
                            # and the known cost of moving from from_loc to to_loc
                            d = costs['find'] + costs['pick'] + map_data.known_cost[from_loc][to_loc]
                            if cost_type == 'pessimistic':
                                d = d + 4000
                            elif cost_type == 'known':
                                d1 = map_data.known_cost[from_loc][cnt_name]
                                d2 = map_data.known_cost[cnt_name][to_loc]
                                d = d1 + d2 + costs['find'] + costs['pick']
                            elif cost_type == 'learned':
                                if from_loc == 'initial_robot_pose':
                                    from_coord = map_data.get_robot_pose()
                                else:
                                    from_coord = learned_data['partial_map'].node_coords[
                                        learned_data['partial_map'].idx_map[from_loc]]
                                if to_loc == 'initial_robot_pose':
                                    to_coord = map_data.get_robot_pose()
                                else:
                                    to_coord = learned_data['partial_map'].node_coords[
                                        learned_data['partial_map'].idx_map[to_loc]]
                                # I need to get the room coords for the from_loc and to_loc
                                # then I can get the expected cost of finding the object
                                # in room level; have it saved per target object
                                if from_loc == 'initial_robot_pose':
                                    # find in which room the robot is at
                                    # which room coord is closest to the robot
                                    from_room_coords = robot_room_coord

                                else:
                                    from_cnt_idx = learned_data['partial_map'].idx_map[from_loc]
                                    room_idx_pos = learned_data['partial_map'].org_edge_index[1].index(from_cnt_idx)
                                    room_idx = learned_data['partial_map'].org_edge_index[0][room_idx_pos]
                                    from_room_coords = learned_data['partial_map'].node_coords[room_idx]

                                if to_loc == 'initial_robot_pose':
                                    to_room_coords = robot_room_coord
                                else:
                                    to_cnt_idx = learned_data['partial_map'].idx_map[to_loc]
                                    room_idx_pos = learned_data['partial_map'].org_edge_index[1].index(to_cnt_idx)
                                    room_idx = learned_data['partial_map'].org_edge_index[0][room_idx_pos]
                                    to_room_coords = learned_data['partial_map'].node_coords[room_idx]

                                # check if the find cost has already been calculated for this object for
                                # these room pairs
                                if (gen_name_child, from_room_coords, to_room_coords) in pre_compute:
                                    intermediate_d = pre_compute[(gen_name_child, from_room_coords, to_room_coords)]
                                else:
                                    intermediate_d, pred_sub = get_expected_cost_of_finding(
                                        learned_data['partial_map'],
                                        learned_data['subgoals'],
                                        child_name,
                                        from_room_coords,  # robot_pose
                                        to_room_coords,  # destination_pose
                                        learned_data['learned_net'],
                                        pred_sub)
                                    pre_compute[(gen_name_child, from_room_coords, to_room_coords)] = intermediate_d
                                if (from_coord, from_room_coords) in grid_cost:
                                    part_from = grid_cost[(from_coord, from_room_coords)]
                                else:
                                    part_from = get_cost(map_data.occupancy_grid, from_coord, from_room_coords)
                                    grid_cost[(from_coord, from_room_coords)] = part_from

                                if (to_coord, to_room_coords) in grid_cost:
                                    part_to = grid_cost[(to_coord, to_room_coords)]
                                else:
                                    part_to = get_cost(map_data.occupancy_grid, to_coord, to_room_coords)
                                    grid_cost[(to_coord, to_room_coords)] = part_to
                                d = costs['find'] + part_from + intermediate_d + part_to

                            init_fluents[('find-cost', child_name, from_loc, to_loc)] = round(d, 4)
                    # or else we can optimistically assume the object is in the nearest
                    # undiscovered location from the to-loc [WILL work on it later!!]
                else:
                    # Object is in the known space
                    init_predicates.append(('is-located', child_name))
                    init_predicates.append(('is-at', child_name, cnt_name))

                    # The expected find cost should be sum of the cost to
                    # cnt_name from the from_loc and then the cost to to_loc
                    # from the cnt_name
                    for from_loc in cnt_names:
                        for to_loc in cnt_names:
                            d1 = map_data.known_cost[from_loc][cnt_name]
                            d2 = map_data.known_cost[cnt_name][to_loc]
                            d = d1 + d2
                            init_fluents[('find-cost', child_name, from_loc, to_loc)] = round(d, 4)

                init_predicates.append(('is-pickable', child_name))
                init_predicates.append(('obj-type', gen_name_child, child_name))
                if gen_name_child == 'egg':
                    # init_predicates.append(f"(is-boilable {child_name})")
                    init_predicates.append(('is-boilable', child_name))
                if gen_name_child in ['pot', 'kettle']:
                    # init_predicates.append(f"(is-boiler {child_name})")
                    init_predicates.append(('is-boiler', child_name))
                if gen_name_child in ['apple', 'tomato', 'potato']:
                    # init_predicates.append(f"(is-peelable {child_name})")
                    init_predicates.append(('is-peelable', child_name))
                if gen_name_child == 'knife':
                    # init_predicates.append(f"(is-peeler {child_name})")
                    init_predicates.append(('is-peeler', child_name))
                if gen_name_child == 'bread':
                    # init_predicates.append(f"(is-toastable {child_name})")
                    init_predicates.append(('is-toastable', child_name))
                if gen_name_child == 'toaster':
                    # init_predicates.append(f"(is-toaster {child_name})")
                    init_predicates.append(('is-toaster', child_name))
                if gen_name_child in ['pot', 'kettle', 'coffeemachine']:
                    init_predicates.append(('is-coffeemaker', child_name))
                if gen_name_child in ['cup', 'mug', 'pot', 'kettle', 'coffeemachine']:
                    init_predicates.append(('is-fillable', child_name))
                if gen_name_child == 'waterbottle':
                    init_predicates.append(('filled-with-water', child_name))
                if gen_name_child == 'coffeegrinds':
                    init_predicates.append(('is-coffeeingredient', child_name))

    for c1 in map_data.known_cost:
        for c2 in map_data.known_cost[c1]:
            if c1 == c2:
                continue
            val = map_data.known_cost[c1][c2]
            init_fluents[('known-cost', c1, c2)] = round(val, 4)

    task = goal_provider(seed, cnt_of_interest, obj_of_interest,
                         objects, goal_type)

    if task is None:
        return None, None
    print(f'Goal: {task}')
    goal = [task]
    # Instead of constructing and returning a PDDL string, make it return a Python dictionary representing the problem components.
    struct = {
        'domain_name': 'indoor',
        'problem_name': 'pick-place-problem',
        'objects': objects,
        'init_predicates': init_predicates,  # List of tuples/strings for non-numeric facts
        'init_fluents': init_fluents,  # Dictionary for numeric fluents
        'goal_states': goal,
        'metric': 'minimize (total-cost)'
    }
    return struct, task
