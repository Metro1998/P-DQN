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

# dic to save vehicles' speed and position info w.r.t its vehicle type
# e.g. vehicles_speed = {'NW_right':'vehicle_id_0', 'vehicle_id_6',
#                        'NS_through':...}
vehicles_speed = {}
vehicles_position = {}

# the edgeID is defined in FW_Inter.edg.xml
# as you may have different definition in your own .edg.xml, change it.
edgeIDs = ['north_in', 'east_in', 'south_in', 'west_in']

# vehicle_types will help to filter the vehicles on the same edge but have different direction.
vehicle_types = ['NW_right', 'NS_through', 'NE_left',
                 'EN_right', 'EW_through', 'ES_left',
                 'SE_right', 'SN_through', 'SW_left',
                 'WS_right', 'WE_through', 'WN_left']

# a cursor to indicate which vehicle type is being selected
type_cursor = 0

for edgeID in edgeIDs:
    traci.edge.subscribeContext(edgeID, tc.CMD_GET_VEHICLETYPE_VARIABLE,
                                [tc.LAST_STEP_VEHICLE_ID_LIST])
    vehicles_on_specific_edge = traci.edge.getContextSubscriptionResults(edgeID)
    for i in range(3):  # on one specific edge, there are three different vehicle types.
        for vehicleID in vehicles_on_specific_edge:
            traci.vehicle.subscribeContext(vehicleID, tc.CMD_GET_VEHICLETYPE_VARIABLE,
                                           [tc.VAR_SPEED, tc.VAR_POSITION])
            traci.vehicle.addSubscriptionFilterVType(vehicle_types[type_cursor])
            vehicles_speed[vehicle_types[type_cursor]] = \
                traci.vehicle.getContextSubscriptionResults(vehicleID)
            vehicles_position[vehicle_types[type_cursor]] = \
                traci.vehicle.getContextSubscriptionResults(vehicleID)
        type_cursor += 1

