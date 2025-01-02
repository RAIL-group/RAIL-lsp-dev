class MDP:
    def __init__(self, state, transitions):
        self.mdp_transitions = transitions
        self.current_state = state
        self.actions = self._get_actions_in_format()

    def transition(self, action):
        outcomes = self.mdp_transitions[self.current_state][action]
        outcome_dict = {MDP(outcome[0], self.mdp_transitions): (outcome[1], outcome[2]) for outcome in outcomes}
        return outcome_dict

    def get_actions(self):
        return list(self.actions[self.current_state])

    def _get_actions_in_format(self):
        actions = {}
        for state, transitions in self.mdp_transitions.items():
            actions[state] = tuple(transitions.keys())
        return actions

    def __hash__(self):
        self.hash_id = hash(self.current_state)
        return self.hash_id

    def __eq__(self, other):
        return self.hash_id == other.hash_id

    def __repr__(self):
        return f'{self.current_state}'
