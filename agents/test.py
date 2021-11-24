import gym
env = gym.make('FreewheelingIntersection-v0')
env.reset()

""" 
import os
import sys
import traci
import traci.constants as tc

if __name__ == "__main__":
    sys.path.append('D:/sumo/tools/')
    # sys.path.append('D:/SUMO/tools')
    # sumoBinary = "/path/to/sumo"
    # sumoBinary = 'D:/SUMO/bin/sumo'
    path = 'envs/sumo/road_network/FW_Inter.sumocfg'
    sumoCmd = ['sumo', '-c', path]
    # create instances
    traci.start(sumoCmd, label='sim1')
    for i in range(50):
        traci.simulationStep()

    # dic to save vehicles' speed and position info w.r.t its vehicle type
    # e.g. vehicles_speed = {'NW_right':'vehicle_id_0', 'vehicle_id_6',
    #                        'NS_through':...}
    vehicles_raw_data = {}

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
    LENGTH_LANE = 234.12

    for edgeID in edgeIDs:
        vehicles_on_specific_edge = []
        traci.edge.subscribe(edgeID, (tc.LAST_STEP_VEHICLE_ID_LIST,))
        # vehicleID is a tuple at this step
        for vehicleID in traci.edge.getSubscriptionResults(edgeID).values():
            for i in range(len(vehicleID)):
                vehicles_on_specific_edge.append(str(vehicleID[i]))

            for ID in vehicles_on_specific_edge:
                tem = []
                traci.vehicle.subscribe(ID, (tc.VAR_TYPE, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                                             tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_TIMELOSS))
                for v in traci.vehicle.getSubscriptionResults(ID).values():
                    tem.append(v)
                print(tem)
                tem[1] = LENGTH_LANE - tem[1]
                # LENGTH_LANE is the length of  lane, gotten from FW_Inter.net.xml.
                # ID:str, vehicle's ID
                # tem[1]:float, the distance between vehicle and lane's stop line.
                # tem[2]:float, speed
                # tem[3]:float, accumulated_waiting_time
                # tem[4]:float, time loss
                if tem[0] not in vehicles_raw_data:
                    vehicles_raw_data[tem[0]] = []
                vehicles_raw_data[tem[0]].append([ID, LENGTH_LANE - tem[1], tem[2], tem[3], tem[4]])

    print(vehicles_raw_data)

"""
