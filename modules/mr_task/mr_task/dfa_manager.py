import spot
import networkx as nx

def _get_single_prop_transtion_dict(aut):
    bdict = aut.get_dict()
    FALSE_BDD = spot.formula_to_bdd(spot.formula("False"), bdict, None)

    all_props = [ap.ap_name() for ap in aut.ap()]
    transition_dict = dict()
    for prop in all_props:
        # Compute the 'condition' object to compare to the edge
        formula = " & ".join([p if p == prop else '!' + p for p in all_props])
        single_prop_condition = spot.formula_to_bdd(spot.formula(formula), bdict, None)

        # Next, loop through edges out of each state & see if
        # any of the 'single_prop_conditions' correspond to the edge
        for state in range(aut.num_states()):
            for edge in aut.out(state):
                if edge.dst == state:
                    continue
                if not (edge.cond & single_prop_condition) == FALSE_BDD:
                    transition_dict[(state, prop)] = edge.dst

    return transition_dict

def _get_downstream_features(transition_dict):
    all_states = set([state for state, _ in transition_dict.keys()])
    all_props = set([prop for _, prop in transition_dict.keys()])
    immediately_useful_props = {
        state: set([prop for prop in all_props if (state, prop) in transition_dict.keys()])
        for state in all_states}

    # This might not even be necessary, but oh well.
    # I build a graph and use depth first search to get all downstream nodes.
    # Then I get all the 'immediately useful' props for each such set.

    # Build a graph to compute downstream nodes
    transitions = nx.DiGraph()
    transitions.add_edges_from(
        ((state_in, state_out)
         for (state_in, prop), state_out in transition_dict.items()
         if not state_in == state_out))

    return {state: set().union(*[immediately_useful_props[sin] for sin, _ in nx.dfs_edges(transitions, state)])
            for state in all_states}



class DFAManager(object):
    def __init__(self, specification, complete=False):
        self.specification = specification

        if complete:
            self.aut = spot.translate(specification, "BA", "complete")
        else:
            self.aut = spot.translate(specification, "BA")

        self.bdict = self.aut.get_dict()
        self.state = self.aut.get_init_state_number()

        self.accepting_states = set(
            t.dst
            for s in range(self.aut.num_states())
            for t in self.aut.out(s)
            if t.acc.count() > 0 and t.dst == t.src
        )

        self.all_props = [ap.ap_name() for ap in self.aut.ap()]

        # Compute single-prop transitions:
        self._transition_dict = _get_single_prop_transtion_dict(self.aut)
        self._useful_props_dict = _get_downstream_features(self._transition_dict)

    def is_accepting_state(self, state):
        return state in self.accepting_states

    def get_useful_props(self):
        return self._useful_props_dict.get(self.state, set())

    def state_id_after_props(self, props):
        state = self.state
        did_change = True
        while did_change:
            did_change = False
            for prop in props:
                upd_state = self._transition_dict.get((state, prop), state)
                # print(state, prop, upd_state)
                if not state == upd_state:
                    state = upd_state
                    did_change = did_change or True

        return state

    def advance(self, props):
        self.state = self.state_id_after_props(props)

    def does_transition_state(self, props):
        return not self.state == self.state_id_after_props(props)
