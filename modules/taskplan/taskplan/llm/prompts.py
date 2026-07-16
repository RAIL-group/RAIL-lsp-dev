import re

def get_room_number(name):
    parts = name.split('|')
    if len(parts) >= 2:
        if parts[1].isdigit():
            return parts[1]
        elif len(parts) >= 3 and parts[2].isdigit():
            return parts[2]
    return None

def parse_goal_to_plain_english(goal_str, problem_struct):
    # Try to find the location and target type
    loc_match = re.search(r'is-at \?\w+\s+([\w|]+)', goal_str)
    type_match = re.search(r'obj-type-(\w+)\s+\?\w+', goal_str)

    if loc_match and type_match:
        target_loc = loc_match.group(1)
        target_type = type_match.group(1)

        # Find all objects in objects list that have this type in their name
        candidates = []
        for obj_type, obj_names in problem_struct.get('objects', {}).items():
            for name in obj_names:
                if target_type in name.lower() and not name.startswith("init_r") and name != target_loc:
                    candidates.append(name)

        if candidates:
            return f"Place a {target_type} at {target_loc} (specifically, one of these candidates: {', '.join(candidates)})"
        else:
            return f"Place a {target_type} at {target_loc}"

    return goal_str

def generate_plain_english_prompt(problem_struct):
    # 1. Get robot location and hand state
    robot_loc = "unknown"
    hand_free = True
    holding_obj = None
    located_objects = []

    for pred in problem_struct.get('init_predicates', []):
        if pred[0] == 'rob-at':
            robot_loc = pred[1]
        elif pred[0] == 'hand-is-free':
            hand_free = True
        elif pred[0] == 'is-holding':
            hand_free = False
            holding_obj = pred[1]
        elif pred[0] == 'is-at':
            located_objects.append(f"- {pred[1]} is at {pred[2]}")

    # 2. Get missing objects
    missing_objects = problem_struct.get('missing_objects', [])

    # 3. Get all locations/objects and group them by room
    locations = []
    for obj_type, obj_names in problem_struct.get('objects', {}).items():
        locations.extend(obj_names)

    goals = problem_struct.get('goal_states', [])
    unexplored_subgoals = problem_struct.get('subgoals', None)

    rooms = {}
    for loc in locations:
        # Check if unexplored or a goal location or the robot's current pose
        is_candidate = True
        if unexplored_subgoals is not None:
            is_goal = False
            for g in goals:
                if loc in str(g):
                    is_goal = True
            is_candidate = (loc in unexplored_subgoals) or is_goal or (loc == robot_loc)

        if not is_candidate:
            continue

        room_num = get_room_number(loc)
        if room_num:
            if room_num not in rooms:
                rooms[room_num] = {"containers": []}
            if loc not in missing_objects:
                rooms[room_num]["containers"].append(loc)

    # Format room layout string
    layout_str = "House layout and candidate containers by rooms:\n"
    for room_num in sorted(rooms.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        layout_str += f"Room {room_num}:\n"
        containers = rooms[room_num]["containers"]
        if containers:
            layout_str += f"  - Candidate search locations/containers: {', '.join(containers)}\n"
    layout_str += "\n"

    if missing_objects:
        layout_str += "Missing objects in the apartment:\n"
        for obj in missing_objects:
            layout_str += f"- {obj}\n"
        layout_str += "\n"

    # Format the prompt state
    state_str = f"Current Robot Location: {robot_loc}\n"
    if hand_free:
        state_str += "Robot's hand is: free\n"
    else:
        state_str += f"Robot is currently holding: {holding_obj}\n"

    if located_objects:
        state_str += "Already located objects:\n" + "\n".join(located_objects) + "\n"
    else:
        state_str += "Already located objects: None\n"

    parsed_goals = [parse_goal_to_plain_english(g, problem_struct) for g in goals]
    goal_str = "Goal:\n" + "\n".join(f"- {g}" for g in parsed_goals) + "\n"

    prompt = f"""You are a task planner for a household robot. Your job is to generate a plan (a sequence of actions) to satisfy the goals from the current state.

Available Actions:
- move(start, end): Move the robot from location 'start' to location 'end'. Precondition: Robot must currently be at 'start'. Start and end must be different locations.
- find(object, location): Search for a missing 'object' at 'location'. Preconditions: Robot must currently be at 'location', the 'object' must be missing, and the robot's hand must be free. Effect: The robot finds and automatically picks up the 'object' (robot is now holding the object). Do NOT call 'pick' after 'find'.
- pick(object, location): Pick up an already located 'object' at 'location'. Preconditions: The 'object' must already be located at 'location', the robot must be at 'location', and the robot's hand must be free.
- place(object, location): Place the held 'object' at 'location'. Precondition: The robot must be holding the 'object' and must currently be at 'location'.

{layout_str}
{state_str}
{goal_str}
Important Rules:
1. CRITICAL: You MUST use the exact, full names of objects and locations as they appear in the available locations and missing objects lists. Do not truncate, alter, or simplify them in any way (e.g. use "diningtable|4|0|0" exactly, do not write "diningtable|4|0" or "kitchen").
2. Only output actions that are logically valid based on preconditions and effects.
3. The 'find' action does not automatically pick up the object, and you still need to call 'pick' after 'find' given that the object was found there during the replanning phase.
4. To find a missing object, you must search for it at a candidate container in the same room where the object is located (e.g. for 'desklamp|surface|4|2', search at 'diningtable|4|0|0' or 'dresser|4|3' in Room 4). You cannot search for it in other rooms.
5. If the robot is already holding a candidate object that satisfies the goal (e.g., you are holding "desklamp|surface|4|2" and the goal requires a desklamp), do not search for any other objects. Instead, generate a plan to move to the goal location and place the held object there.
6. Precondition constraint: You can only search (use 'find') at a location if the robot is currently at that location. If you want to search at a container, you must first 'move' to that container, and then 'find' at that exact same container (e.g., move to 'diningtable|7|1|0', then find at 'diningtable|7|1|0'). Do not mix up different container names.
7. If a candidate object that satisfies the goal is already located (e.g., 'desklamp|surface|4|2 is at diningtable|4|0|0' is in the 'Already located objects' list), do not search for any other objects. Instead, generate a plan to pick up that located object, move to the goal location, and place it there.
8. To pick up an already located object, you must be at the exact same location where the object is. You must call 'pick' before moving away from that location.

Please output a plan that achieves the goals from the current state.
You must return the plan as a JSON object containing a "plan" key, which is a list of actions. Each action has an "action" name (matching the action names above) and an "args" list of arguments (matching the exact names above).

Example format:
{{
  "plan": [
    {{
      "action": "move",
      "args": ["initial_robot_pose", "countertop|1|0"]
    }},
    {{
      "action": "find",
      "args": ["bread|1", "countertop|1|0"]
    }},
    {{  # Given that the object is found there it still needs to pick during replan
      "action": "pick",
      "args": ["bread|1", "countertop|1|0"]
    }},
    {{
      "action": "move",
      "args": ["countertop|1|0", "diningtable|2|0"]
    }},
    {{
      "action": "place",
      "args": ["bread|1", "diningtable|2|0"]
    }}
  ]
}}

Generate the plan to satisfy the goals. Do not explain the reasoning, just output the JSON object.
"""
    return prompt

def generate_prompt(domain_pddl, problem_pddl=None, missing_objects=None):
    # if isinstance(domain_pddl, dict):
    #     return generate_plain_english_prompt(domain_pddl)

    missing_str = ""
    if missing_objects:
        missing_str = "Here is the list of missing objects whose locations are currently unknown (you must search for them using the 'find' action at a candidate location):\n"
        for obj in missing_objects:
            missing_str += f"- {obj}\n"
        missing_str += "\n"

    return f"""You are a task planner for a household robot. Your job is to generate a plan (a sequence of actions) to satisfy the goals defined in the PDDL problem below, using the actions defined in the PDDL domain.

Here is the PDDL Domain:
```pddl
{domain_pddl}
```

Here is the PDDL Problem representing the current state of the world and the goals:
```pddl
{problem_pddl}
```

{missing_str}

Important Rules:
1. CRITICAL: You MUST use the exact, full names of objects and locations as they appear in the available locations and missing objects lists. Do not truncate, alter, or simplify them in any way (e.g. use "diningtable|4|0|0" exactly, do not write "diningtable|4|0" or "kitchen").
2. Only output actions that are logically valid based on preconditions and effects.
3. The 'find' action does not automatically pick up the object, and you still need to call 'pick' after 'find' given that the object was found there during the replanning phase.
4. To find a missing object, you must search for it at a candidate container in the same room where the object is located (e.g. for 'desklamp|surface|4|2', search at 'diningtable|4|0|0' or 'dresser|4|3' in Room 4). You cannot search for it in other rooms.
5. If the robot is already holding a candidate object that satisfies the goal (e.g., you are holding "desklamp|surface|4|2" and the goal requires a desklamp), do not search for any other objects. Instead, generate a plan to move to the goal location and place the held object there.
6. Precondition constraint: You can only search (use 'find') at a location if the robot is currently at that location. If you want to search at a container, you must first 'move' to that container, and then 'find' at that exact same container (e.g., move to 'diningtable|7|1|0', then find at 'diningtable|7|1|0'). Do not mix up different container names.
7. If a candidate object that satisfies the goal is already located (e.g., 'desklamp|surface|4|2 is at diningtable|4|0|0' is in the 'Already located objects' list), do not search for any other objects. Instead, generate a plan to pick up that located object, move to the goal location, and place it there.
8. To pick up an already located object, you must be at the exact same location where the object is. You must call 'pick' before moving away from that location.

Please output a plan that achieves the goals from the current state.
You must return the plan as a JSON object containing a "plan" key, which is a list of actions. Each action has an "action" name (matching the action names above) and an "args" list of arguments (matching the exact names above).

Example format:
{{
  "plan": [
    {{
      "action": "move",
      "args": ["initial_robot_pose", "countertop|1|0"]
    }},
    {{
      "action": "find",
      "args": ["bread|1", "countertop|1|0"]
    }},
    {{  # Given that the object is found there it still needs to pick during replan
      "action": "pick",
      "args": ["bread|1", "countertop|1|0"]
    }},
    {{
      "action": "move",
      "args": ["countertop|1|0", "diningtable|2|0"]
    }},
    {{
      "action": "place",
      "args": ["bread|1", "diningtable|2|0"]
    }}
  ]
}}

Generate the plan to satisfy the goals in the PDDL problem. Do not explain the reasoning, just output the JSON object.
"""

def get_system_prompt(use_thinking=False):
    base = (
        "You are an expert robot task planner. "
        "You must generate a plan to achieve the goal state. "
        "Your response MUST be a JSON object containing a single 'plan' key mapping to a list of actions. "
        "Do NOT write any introduction, greeting, or conversational filler. Output only the raw JSON plan."
    )
    if use_thinking:
        return f"{base} <|think|> Think step by step about state transitions and action preconditions before generating the JSON plan."
    else:
        return base
