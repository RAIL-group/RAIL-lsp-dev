import pytest
from unittest.mock import patch, MagicMock
from taskplan.llm import prompts, plan_parser, validator
from taskplan.planners import llm_planner

def test_prompt_generation():
    domain = "(define (domain test) ...)"
    problem = "(define (problem test) ...)"
    prompt = prompts.generate_prompt(domain, problem)
    assert domain in prompt
    assert problem in prompt

def test_plain_english_prompt_generation():
    problem_struct = {
        'objects': {
            'location': ['diningtable|4|0|0', 'dresser|4|3'],
            'item': ['desklamp|surface|4|2', 'book|surface|3|1']
        },
        'init_predicates': [
            ('rob-at', 'diningtable|4|0|0'),
            ('hand-is-free',),
        ],
        'missing_objects': ['desklamp|surface|4|2', 'book|surface|3|1'],
        'goal_states': [],
    }

    prompt = prompts.generate_prompt(problem_struct)

    # Check that missing objects are not listed under room descriptions
    assert "Missing objects belonging to this room" not in prompt

    # Check that missing objects are listed in the apartment-wide section
    assert "Missing objects in the apartment:" in prompt
    assert "- desklamp|surface|4|2" in prompt
    assert "- book|surface|3|1" in prompt

def test_plan_parser():
    # Valid plan JSON
    valid_json = '{"plan": [{"action": "move", "args": ["a", "b"]}, {"action": "find", "args": ["c", "b"]}]}'
    actions = plan_parser.parse_plan(valid_json)
    assert len(actions) == 2
    assert actions[0].name == "move"
    assert actions[0].args == ("a", "b")
    assert actions[1].name == "find"
    assert actions[1].args == ("c", "b")

    # JSON with markdown tags
    md_json = '```json\n{"plan": [{"action": "pick", "args": ["x", "y"]}]}\n```'
    actions = plan_parser.parse_plan(md_json)
    assert len(actions) == 1
    assert actions[0].name == "pick"

    # Invalid JSON
    with pytest.raises(ValueError):
        plan_parser.parse_plan("invalid json")

    # Missing plan key
    with pytest.raises(ValueError):
        plan_parser.parse_plan('{"not_plan": []}')

def test_validator():
    problem_struct = {
        'objects': {
            'location': ['loc1', 'loc2'],
            'item': ['item1', 'item2']
        },
        'init_predicates': [
            ('rob-at', 'loc1'),
            ('hand-is-free',),
            ('is-pickable', 'item1'),
            ('is-located', 'item1'),
            ('is-at', 'item1', 'loc1'),
            ('restrict-move-to', 'loc2'),
        ]
    }

    val = validator.PDDLStateValidator(problem_struct)

    # Valid pick action
    assert val.validate_and_apply(plan_parser.Action(name='pick', args=('item1', 'loc1'))) == True
    assert ('is-holding', 'item1') in val.facts
    assert ('hand-is-free',) not in val.facts

    # Try picking again (invalid because hand is not free)
    assert val.validate_and_apply(plan_parser.Action(name='pick', args=('item1', 'loc1'))) == False

    # Valid place action
    assert val.validate_and_apply(plan_parser.Action(name='place', args=('item1', 'loc1'))) == True
    assert ('is-at', 'item1', 'loc1') in val.facts
    assert ('hand-is-free',) in val.facts

    # Move to loc2 (invalid because loc2 is restricted)
    assert val.validate_and_apply(plan_parser.Action(name='move', args=('loc1', 'loc2'))) == False

    # Move to loc3 (invalid because loc3 is not in objects)
    assert val.validate_and_apply(plan_parser.Action(name='move', args=('loc1', 'loc3'))) == False

def test_llm_planner_integration():
    domain_pddl = "(domain definition)"
    problem_struct = {
        'problem_name': 'mock-problem',
        'domain_name': 'mock-domain',
        'goal_states': [],
        'objects': {
            'location': ['loc1', 'loc2'],
            'item': ['item1']
        },
        'init_predicates': [
            ('rob-at', 'loc1'),
            ('hand-is-free',),
            ('is-pickable', 'item1'),
            ('is-located', 'item1'),
            ('is-at', 'item1', 'loc1'),
        ],
        'init_fluents': {
            ('known-cost', 'loc1', 'loc2'): 5.5
        }
    }
    partial_map = MagicMock()
    args = MagicMock()
    args.llm_base_url = "http://localhost:11434/api/chat"
    args.llm_model = "gemma-4-e4b"
    args.llm_use_thinking = False

    # Mock client response
    # Actions: pick (valid), find (invalid because hand is not free)
    mock_response = '{"plan": [{"action": "pick", "args": ["item1", "loc1"]}, {"action": "find", "args": ["item1", "loc1"]}]}'

    with patch('taskplan.llm.client.OllamaClient.query', return_value=mock_response):
        actions, cost = llm_planner.solve_with_llm(domain_pddl, problem_struct, partial_map, args)

        # Actions should truncate to just [pick] because find preconditions fail
        assert len(actions) == 1
        assert actions[0].name == "pick"
