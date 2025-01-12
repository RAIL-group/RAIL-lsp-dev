from taskplan.pddl.helper import generate_pddl_problem, goal_provider, \
    get_expected_cost_of_finding
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
    init_states = [
        '(= (total-cost) 0)',
        '(restrict-move-to initial_robot_pose)',
        '(hand-is-free)',
        '(rob-at initial_robot_pose)'  # , '(is-fillable coffeemachine)'
    ]
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

                if cnt_name in unvisited:
                    # Object is in the unknown space
                    init_states.append(f"(not (is-located {child_name}))")

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
                                d = d + 10000
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
                                if (child_name, from_room_coords, to_room_coords) in pre_compute:
                                    intermediate_d = pre_compute[(child_name, from_room_coords, to_room_coords)]
                                else:
                                    intermediate_d = get_expected_cost_of_finding(
                                        learned_data['partial_map'],
                                        learned_data['subgoals'],
                                        child_name,
                                        from_room_coords,  # robot_pose
                                        to_room_coords,  # destination_pose
                                        learned_data['learned_net'],
                                        learned_data['tsp_cost'])
                                    pre_compute[(child_name, from_room_coords, to_room_coords)] = intermediate_d
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
                                d = part_from + intermediate_d + part_to

                            init_states.append(f"(= (find-cost {child_name} {from_loc} {to_loc}) {d})")
                    # or else we can optimistically assume the object is in the nearest
                    # undiscovered location from the to-loc [WILL work on it later!!]
                else:
                    # Object is in the known space
                    init_states.append(f"(is-located {child_name})")
                    init_states.append(f"(is-at {child_name} {cnt_name})")

                    # The expected find cost should be sum of the cost to
                    # cnt_name from the from_loc and then the cost to to_loc
                    # from the cnt_name
                    for from_loc in cnt_names:
                        for to_loc in cnt_names:
                            d1 = map_data.known_cost[from_loc][cnt_name]
                            d2 = map_data.known_cost[cnt_name][to_loc]
                            d = d1 + d2
                            init_states.append(f"(= (find-cost {child_name} {from_loc} {to_loc}) {d})")

    #             if 'pickable' in child and child['pickable'] == 1:
                init_states.append(f"(is-pickable {child_name})")
                init_states.append(f"(obj-type-{gen_name_child} {child_name})")
                if gen_name_child == 'egg':
                    init_states.append(f"(is-boilable {child_name})")
                if gen_name_child in ['pot', 'kettle']:
                    init_states.append(f"(is-boiler {child_name})")
                if gen_name_child in ['apple', 'tomato', 'potato']:
                    init_states.append(f"(is-peelable {child_name})")
                if gen_name_child == 'knife':
                    init_states.append(f"(is-peeler {child_name})")
                if gen_name_child == 'bread':
                    init_states.append(f"(is-toastable {child_name})")
                if gen_name_child == 'toaster':
                    init_states.append(f"(is-toaster {child_name})")
                if gen_name_child in ['pot', 'kettle', 'coffeemachine']:
                    init_states.append(f"(is-coffeemaker {child_name})")
                if gen_name_child in ['cup', 'mug', 'pot', 'kettle', 'coffeemachine']:
                    init_states.append(f"(is-fillable {child_name})")
                if gen_name_child == 'waterbottle':
                    init_states.append(f"(filled-with-water {child_name})")
                if gen_name_child == 'coffeegrinds':
                    init_states.append(f"(is-coffeeingredient {child_name})")

    for c1 in map_data.known_cost:
        for c2 in map_data.known_cost[c1]:
            if c1 == c2:
                continue
            val = map_data.known_cost[c1][c2]
            init_states.append(
                f"(= (known-cost {c1} {c2}) {val})"
            )

    # task = get_goals(seed, cnt_of_interest, obj_of_interest)
    task = goal_provider(seed, cnt_of_interest, obj_of_interest,
                         objects, goal_type)

    if task is None:
        return None, None
    print(f'Goal: {task}')
    goal = [task]
    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='indoor',
        problem_name='pick-place-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL, task
