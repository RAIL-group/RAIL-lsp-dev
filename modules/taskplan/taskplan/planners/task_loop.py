import time
import re
import taskplan
import taskplan.planners.llm_planner

from pddlstream.algorithms.search import solve_from_pddl
from taskplan.utilities.utils import get_container_pose
from taskplan.planners.planner import ClosestActionPlanner, LearnedPlanner, KnownPlanner


def parse_sexp(s):
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s.split()
    def read_from_tokens(tokens):
        if len(tokens) == 0:
            raise SyntaxError('unexpected EOF')
        token = tokens.pop(0)
        if token == '(':
            l = []
            while tokens[0] != ')':
                l.append(read_from_tokens(tokens))
            tokens.pop(0) # pop ')'
            return l
        elif token == ')':
            raise SyntaxError('unexpected )')
        else:
            return token
    return read_from_tokens(tokens)

def evaluate_goal(goal_exp, state_preds, all_objs, env):
    if goal_exp[0] == 'and':
        return all(evaluate_goal(sub, state_preds, all_objs, env) for sub in goal_exp[1:])
    elif goal_exp[0] == 'or':
        return any(evaluate_goal(sub, state_preds, all_objs, env) for sub in goal_exp[1:])
    elif goal_exp[0] == 'exists':
        var_name = goal_exp[1][0]
        body = goal_exp[2]
        for obj in all_objs:
            new_env = env.copy()
            new_env[var_name] = obj
            if evaluate_goal(body, state_preds, all_objs, new_env):
                return True
        return False
    elif goal_exp[0] == 'not':
        return not evaluate_goal(goal_exp[1], state_preds, all_objs, env)
    else:
        pred_name = goal_exp[0]
        args = [env.get(arg, arg) for arg in goal_exp[1:]]
        if pred_name.startswith('obj-type-'):
            expected_type = pred_name[len('obj-type-'):].lower()
            return expected_type in args[0].lower()
        target_pred = tuple([pred_name] + args)
        return target_pred in state_preds

def is_goal_satisfied(problem_struct):
    state_preds = set([tuple(p) for p in problem_struct.get('init_predicates', [])])
    all_objs = set()
    for objs in problem_struct.get('objects', {}).values():
        all_objs.update(objs)
        
    for goal_str in problem_struct.get('goal_states', []):
        try:
            goal_exp = parse_sexp(goal_str)
            if not evaluate_goal(goal_exp, state_preds, all_objs, {}):
                return False
        except Exception as e:
            print(f"Goal evaluation failed for {goal_str}: {e}")
            return False
    return True


def run(plan, pddl, partial_map, init_robot_pose, args):
    costs = taskplan.utilities.utils.get_action_costs()
    executed_actions = []
    robot_poses = [init_robot_pose]
    action_cost = 0
    replan_count = 0
    max_replans = len(pddl['subgoals'])
    while plan:
        for action_idx, action in enumerate(plan):
            # Break loop at the end of plan
            if action_idx == len(plan) - 1:
                plan = []
            executed_actions.append(action)
            if action.name != 'move':
                action_cost += costs[action.name]

            if action.name == 'pour-water':
                pour_from = action.args[0]
                pour_to = action.args[1]
                # Update problem for pour-water action.
                # (filled-with-water ?pour_to)
                # (not (filled-with-water ?pour_from))
                # (not (ban-move))
                taskplan.pddl.helper.update_problem_pourwater(
                    pddl['problem_struct'], pour_from, pour_to)
            elif action.name == 'pour-coffee':
                pour_from = action.args[0]
                pour_to = action.args[1]
                # Update problem for pour-coffee action.
                # (filled-with-coffee ?pour_to)
                # (not (filled-with-coffee ?pour_from))
                # (not (ban-move))
                taskplan.pddl.helper.update_problem_pourcoffee(
                    pddl['problem_struct'], pour_from, pour_to)
            elif action.name == 'make-coffee':
                receptacle = action.args[1]
                # Update problem for make-coffee action.
                # (filled-with-coffee ?receptacle)
                # (not (filled-with-water ?receptacle))
                # (not (ban-move))
                taskplan.pddl.helper.update_problem_makecoffee(
                    pddl['problem_struct'], receptacle)
            elif action.name == 'move':
                move_start = action.args[0]
                ms_pose = get_container_pose(move_start, partial_map)
                if ms_pose is None:
                    ms_pose = init_robot_pose
                move_end = action.args[1]
                me_pose = get_container_pose(move_end, partial_map)
                if me_pose is None:
                    me_pose = init_robot_pose

                # Update problem for move action.
                # (rob-at move_end)
                robot_poses.append(me_pose)
                taskplan.pddl.helper.update_problem_move(
                    pddl['problem_struct'], move_end)
            elif action.name == 'pick':
                object_name = action.args[0]
                pick_at = action.args[1]
                pick_pose = get_container_pose(pick_at, partial_map)
                if pick_pose is None:
                    pick_pose = init_robot_pose
                # Update problem for pick action.
                # (not (hand-is-free))
                # (not (is-at object location))
                # (is holding object)
                taskplan.pddl.helper.update_problem_pick(
                    pddl['problem_struct'], object_name, pick_at)
            elif action.name == 'place':
                object_name = action.args[0]
                place_at = action.args[1]
                place_pose = get_container_pose(place_at, partial_map)
                if place_pose is None:
                    place_pose = init_robot_pose
                # Update problem for place action.
                # (hand-is-free)
                # (is-at object location)
                # (not (is holding object))
                taskplan.pddl.helper.update_problem_place(
                    pddl['problem_struct'], object_name, place_at)
            elif action.name == 'boil':
                obj1_name = action.args[0]
                # Update problem for boil action.
                # (is-boiled obj1_name)
                taskplan.pddl.helper.update_problem_boil(
                    pddl['problem_struct'], obj1_name)
            elif action.name == 'peel':
                obj1_name = action.args[0]
                # Update problem for peel action.
                # (is-peeled obj1_name)
                taskplan.pddl.helper.update_problem_peel(
                    pddl['problem_struct'], obj1_name)
            elif action.name == 'toast':
                obj1_name = action.args[0]
                # Update problem for toast action.
                # (is-toasted obj1_name)
                taskplan.pddl.helper.update_problem_toast(
                    pddl['problem_struct'], obj1_name)
            elif action.name == 'find':
                obj_name = action.args[0]
                obj_idx = partial_map.idx_map[obj_name]
                if len(action.args) == 3:
                    find_start = action.args[1]
                    find_end = action.args[2]
                else:
                    find_start = action.args[1]
                    find_end = action.args[1]
                fs_pose = get_container_pose(find_start, partial_map)
                if fs_pose is None:
                    fs_pose = init_robot_pose
                fe_pose = get_container_pose(find_end, partial_map)
                if fe_pose is None:
                    fe_pose = init_robot_pose

                # Initialize the partial map
                partial_map.target_obj = obj_idx

                is_llm = getattr(args, 'planner_backend', 'pddl') == 'llm'
                if is_llm:
                    explored_loc = partial_map.idx_map[find_end]
                    pddl['subgoals'].remove(explored_loc)
                else:
                    # Over here initiate the planner
                    if 'greedy' in args.logfile_name:
                        planner = ClosestActionPlanner(args, partial_map,
                                                       destination=fe_pose)
                    elif 'oracle' in args.logfile_name:
                        planner = KnownPlanner(args, partial_map,
                                               destination=fe_pose)
                    else:
                        planner = LearnedPlanner(args, partial_map, verbose=True,
                                                 destination=fe_pose)
                    # Initiate planning loop but run for a step
                    planning_loop = taskplan.planners.planning_loop.PlanningLoop(
                        partial_map=partial_map, robot=fs_pose,
                        destination=fe_pose, args=args, verbose=True)

                    planning_loop.subgoals = pddl['subgoals'].copy()
                    explored_loc = None

                    for counter, step_data in enumerate(planning_loop):
                        # Update the planner objects
                        s_time = time.time()
                        planner.update(
                            step_data['graph'],
                            step_data['subgoals'],
                            step_data['robot_pose'])
                        print(f"Time taken to update: {time.time() - s_time}")

                        # Compute the next subgoal and set to the planning loop
                        s_time = time.time()
                        chosen_subgoal = planner.compute_selected_subgoal()
                        print(f"Time taken to choose subgoal: {time.time() - s_time}")
                        planning_loop.set_chosen_subgoal(chosen_subgoal)

                        explored_loc = chosen_subgoal.value
                        pddl['subgoals'].remove(chosen_subgoal.value)
                        break  # Run the loop only exploring one containers

                # Get which container was chosen to explore
                # Get the objects that are connected to that container
                idx2assetID = {partial_map.idx_map[assetID]: assetID for assetID in partial_map.idx_map}

                connection_idx = [
                    partial_map.org_edge_index[1][idx]
                    for idx, value in enumerate(partial_map.org_edge_index[0])
                    if value == explored_loc
                ]
                found_objects = [
                    idx2assetID[con_idx]
                    for con_idx in connection_idx
                ]
                found_at = idx2assetID[explored_loc]

                # Update problem for find action.
                robot_poses.append(partial_map.container_poses[explored_loc])
                is_llm = getattr(args, 'planner_backend', 'pddl') == 'llm'
                taskplan.pddl.helper.update_problem_find(
                    pddl['problem_struct'], found_objects, found_at, find_start)
                # call the next update only for learned cost_type to update
                # the new find costs after exploring a container
                if is_llm or args.cost_type == 'learned':
                    pddl['problem_struct']['missing_objects'] = [
                        obj for obj in pddl['problem_struct']['missing_objects']
                        if obj not in found_objects
                    ]
                if args.cost_type == 'learned':
                    taskplan.pddl.helper.update_find_costs(
                        pddl['problem_struct'], partial_map,
                        args.network_file, pddl['subgoals'],
                        init_robot_pose, args.robot_room_coord
                    )
                pddl['problem'] = taskplan.pddl.helper.\
                    generate_pddl_problem_from_struct(pddl['problem_struct'])

                # Finally replan
                print('Replanning .. .. ..')
                if is_llm:
                    pddl['problem_struct']['subgoals'] = [
                        name for name, idx in partial_map.idx_map.items()
                        if idx in pddl['subgoals']
                    ]
                    pddl['problem_struct']['init_predicates'] = [
                        pred for pred in pddl['problem_struct']['init_predicates']
                        if pred[0] not in ('ban-move', 'ban-find')
                    ]
                    plan, cost, has_error = taskplan.planners.llm_planner.solve_with_llm(
                        pddl['domain'], pddl['problem_struct'], partial_map, args)
                    if has_error:
                        taskplan.utilities.utils.terminate_experiment(args, "LLM hallucinated invalid action arguments during replan.")
                else:
                    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                                                 planner=pddl['planner'], max_planner_time=240)
                cost_str = taskplan.utilities.utils.get_cost_string(args)
                taskplan.utilities.utils.check_replan_validity(plan, args, cost_str)
                replan_count += 1
                if replan_count > max_replans:
                    taskplan.utilities.utils.terminate_experiment(args, f"Max replan count ({max_replans}) exceeded. Terminating.")
                break

        if not plan:
            is_llm = getattr(args, 'planner_backend', 'pddl') == 'llm'
            if is_llm and not is_goal_satisfied(pddl['problem_struct']):
                taskplan.utilities.utils.terminate_experiment(args, "Plan ended but goal not satisfied and no valid find action to replan from. Terminating experiment.")

    return executed_actions, robot_poses, action_cost
