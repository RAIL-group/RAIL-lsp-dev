from enum import Enum

EventOutcome = Enum('EventOutcome', ['BLOCK', 'TRAV','CHANCE'])
RobotType = Enum('RobotType', ['Ground', 'Drone'])
BLOCK_COST = 3e1
STUCK_COST = 15.0 #30.0
VEL_RATIO = 2.0
APPROX_TIME = 1e-5
TRAV_LEVEL = 0.45
REVISIT_PEN = 10.0