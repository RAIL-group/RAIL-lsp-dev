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



# actions = {
#     'S': {'A', 'B'}, 'S2': {'B'}, 'S4': {'A'},
#     'S1': {}, 'S3': {}, 'S21': {}, 'S22': {}, 'S41': {}, 'S42': {}
# }
'''
States: S -> S1 -> S2
Actions: S: A leads to S1 with cost 10
         S1: B leads to S2 with cost 5
         S2: None
Goal: Choose action A->B with total cost 15
'''
# class DeterministicMDP(MDP):
#     def __init__(self, state='S'):
#         super().__init__(state)
#         self.transitions = {
#             'S': {'A': [('S1', 1.0, 10)]},
#             'S1': {'B': [('S2', 1.0, 5)]},
#             'S2': {}
#         }
