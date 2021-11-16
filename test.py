import os
import sys
import traci
import traci.constants as tc

sys.path.append('D:/SUMO/tools')
# sumoBinary = "/path/to/sumo"
sumoBinary = 'D:/SUMO/bin/sumo'
sumoCmd = [sumoBinary, '-c', 'FW_Inter.sumocfg']

# create instances
traci.start(sumoCmd, label='sim1')
traci.