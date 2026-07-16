import json
import re
from collections import namedtuple

Action = namedtuple('Action', ['name', 'args'])

def parse_plan(response_text):
    # Strip code block decorators if the model outputs markdown wrapped JSON
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\nRaw response:\n{response_text}"
        )

    if not isinstance(data, dict) or "plan" not in data:
        raise ValueError(
            f"LLM response JSON is missing the 'plan' key.\nRaw response:\n{response_text}"
        )

    plan_list = data["plan"]
    if not isinstance(plan_list, list):
        raise ValueError(
            f"Expected 'plan' to be a list, got {type(plan_list)}.\nRaw response:\n{response_text}"
        )

    actions = []
    for idx, item in enumerate(plan_list):
        if not isinstance(item, dict) or "action" not in item or "args" not in item:
            raise ValueError(
                f"Action entry at index {idx} must be a dict containing 'action' and 'args': {item}"
            )
        action_name = str(item["action"]).strip()
        action_args = tuple(str(arg).strip() for arg in item["args"])
        actions.append(Action(name=action_name, args=action_args))

    return actions
