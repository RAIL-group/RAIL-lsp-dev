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


def test_mrtask_dfa_advance():
    specification = 'F Knife & F Pillow'
    dfa = DFAManager(specification)
    objects_found = ()
    dfa.advance(objects_found)
    assert 'Knife' in dfa.get_useful_props()
    assert 'Pillow' in dfa.get_useful_props()

    objects_found = ('Knife', 'Pillow',)
    print(f'{objects_found=}')
    dfa.advance(objects_found)
    assert not 'Knife' in dfa.get_useful_props()
    assert not 'Pillow' in dfa.get_useful_props()


def test_mrtask_dfa_ordered_specification():
    specification = '(!foo U bar) & (!bar U qux) & (F foo)'
    dfa = DFAManager(specification)

    assert 'foo' in dfa.get_useful_props()
    assert 'bar' in dfa.get_useful_props()
    assert 'qux' in dfa.get_useful_props()

    assert not dfa.does_transition_state(('foo',))
    assert not dfa.does_transition_state(('bar',))
    assert dfa.does_transition_state(('qux',))

    dfa.advance(('qux',))
    assert not dfa.does_transition_state(('foo',))
    assert dfa.does_transition_state(('bar',))

    dfa.advance(('bar',))
    assert dfa.does_transition_state(('foo',))

    dfa.advance(('foo',))
    assert dfa.has_reached_accepting_state()
