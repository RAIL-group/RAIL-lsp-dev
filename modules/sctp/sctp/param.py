from enum import Enum

EventOutcome = Enum('EventOutcome', ['BLOCK', 'TRAV','CHANCE'])
RobotType = Enum('RobotType', ['Ground', 'Drone'])
MAX_EDGE_LENGTH = 8.0
MIN_EDGE_LENGTH = 3.0
MAX_UAV_ACTION = 1
IV_SAMPLE_SIZE = 50
BLOCK_COST = 0.0
STUCK_COST = 0.0
NOWAY_PEN = 200.0
VEL_RATIO = 2.0
APPROX_TIME = 1e-5
TRAV_LEVEL = 0.6
REVISIT_PEN = 0.0
# REVISIT_PEN = 6.0
ADD_IV = True