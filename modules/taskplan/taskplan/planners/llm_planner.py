import taskplan.pddl.helper
import taskplan.utilities.utils
from taskplan.llm.client import OllamaClient
from taskplan.llm import prompts
from taskplan.llm import plan_parser
from taskplan.llm.validator import PDDLStateValidator


def solve_with_llm(domain_pddl, problem_struct, partial_map, args):
    # 1. Generate the PDDL problem string representation without verbose known-cost fluents
    import copy
    clean_struct = copy.deepcopy(problem_struct)
    if 'init_fluents' in clean_struct:
        clean_struct['init_fluents'] = {
            k: v for k, v in clean_struct['init_fluents'].items()
            if k[0] != 'known-cost'
        }
    problem_pddl = taskplan.pddl.helper.generate_pddl_problem_from_struct(clean_struct)

    # 2. Extract client settings from arguments with fallbacks
    base_url = getattr(args, 'llm_base_url', "http://localhost:11434/api/chat")
    model = getattr(args, 'llm_model', "gemma-4-e4b")
    use_thinking = getattr(args, 'llm_use_thinking', True)

    # 3. Build prompts and query the Ollama API
    prompt = prompts.generate_prompt(domain_pddl, problem_struct, clean_struct.get('missing_objects', []))
    system_prompt = prompts.get_system_prompt(use_thinking=use_thinking)

    client = OllamaClient(base_url=base_url, model=model)
    try:
        response_text = client.query(
            prompt=prompt,
            response_format={"type": "json_object"},
            system_prompt=system_prompt
        )
    except ConnectionError as e:
        print(f"LLM query failed: {e}")
        return [], 0.0, "LLM client timed out or connection failed."

    # print("--- LLM PROMPT ---", flush=True)
    # print(prompt, flush=True)
    # print("--- SYSTEM PROMPT ---", flush=True)
    # print(system_prompt, flush=True)
    print("--- LLM RESPONSE ---", flush=True)
    print(response_text, flush=True)
    print("--------------------", flush=True)



    # 4. Parse the resulting JSON plan
    try:
        actions = plan_parser.parse_plan(response_text)
    except Exception as e:
        print(f"Failed to parse LLM plan: {e}")
        actions = []

    if not actions:
        from taskplan.planners.task_loop import is_goal_satisfied
        if is_goal_satisfied(problem_struct):
            print("Goal is already satisfied. No actions needed.")
            return [], 0.0, None
        else:
            print("LLM returned an empty or invalid plan, but the goal is not satisfied.")
            return [], 0.0, "LLM failed to generate a valid plan for the unsatisfied goal."

    # 5. Forward-simulate and validate the actions, truncating at the first invalid
    # action or immediately after the first find action.
    validator = PDDLStateValidator(problem_struct)
    validated_actions = []
    error_msg = None
    for action in actions:
        if validator.validate_and_apply(action):
            validated_actions.append(action)
            if action.name.lower() == 'find':
                break
        else:
            print(f"Truncating plan due to invalid LLM action: {action}")
            error_msg = "LLM hallucinated invalid action arguments in initial plan."
            break

    # 6. Sum the action costs for the validated plan prefix
    costs = taskplan.utilities.utils.get_action_costs()
    total_cost = 0.0
    for action in validated_actions:
        name = action.name.lower()
        if name == 'move':
            if len(action.args) == 2:
                start, end = action.args
                cost_val = problem_struct.get('init_fluents', {}).get(('known-cost', start, end), 0.0)
                total_cost += float(cost_val)
        else:
            total_cost += float(costs.get(name, 0.0))

    print(f"Validated LLM plan prefix: {validated_actions} with cost: {total_cost}")
    return validated_actions, total_cost, error_msg
