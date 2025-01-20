import pytest
from mr_task import DFAManager


@pytest.mark.parametrize('specification', [
    'F objA & F objB', 'F objA | F objB',])
def test_mrtask_dfa_manager(specification):
    dfa = DFAManager(specification)
    # assert len(dfa.all_props) == 2
    assert 'objA' in dfa.all_props
    assert 'objB' in dfa.all_props

    assert dfa.does_transition_state(('objA',))
    assert dfa.does_transition_state(('objB',))
    assert dfa.does_transition_state(('objA', 'objB'))
    assert dfa.does_transition_state(('objA', 'objC'))
    assert not dfa.does_transition_state(('objC',))
    assert dfa.is_accepting_state(dfa.state_id_after_props(('objA', 'objB')))

    # Next, we check to see which props are 'useful'.
    # This can be used to filter out no-longer-useful actions
    dfa = DFAManager(specification)
    assert 'objA' in dfa.get_useful_props()
    assert 'objB' in dfa.get_useful_props()
    assert not 'objC' in dfa.get_useful_props()
    dfa.advance(('objA',))
    assert not 'objA' in dfa.get_useful_props()
    dfa.advance(('objB',))
    assert not 'objB' in dfa.get_useful_props()
    assert dfa.is_accepting_state(dfa.state)
