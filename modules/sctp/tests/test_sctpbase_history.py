import pytest
from sctp import base_pomdpstate

def test_sctpbase_history():
   start = 1
   im_node1 = 2
   im_node2 = 3
   goal = 4
   history1 = base_pomdpstate.History()
   history2 = base_pomdpstate.History()
   history3 = base_pomdpstate.History()
   EnventOutcome = base_pomdpstate.EventOutcome

   history1.add_history(action=im_node1, start=start, prob=0.5, outcome=EnventOutcome.BLOCK)
   history2.add_history(action=im_node1, start=start, prob=0.5, outcome=EnventOutcome.BLOCK)
   history3.add_history(action=im_node1, start=start, prob=0.5, outcome=EnventOutcome.TRAV)
   assert history1 == history2
   assert history1 != history3

   history1.add_history(action=goal, start=im_node1, prob=0.5, outcome=EnventOutcome.TRAV)
   history2.add_history(action=goal, start=im_node1, prob=0.5, outcome=EnventOutcome.TRAV)
   assert history1 == history2

   history1.add_history(action=goal, start=im_node1, prob=0.5, outcome=EnventOutcome.TRAV)
   history2.add_history(action=start, start=im_node1, prob=0.5, outcome=EnventOutcome.TRAV)

   assert history1 != history2