# @author Metro
# @time 2021/11/11
# happy to be single!
# 暂时不考虑加入参数
# 先实现自己的状态空间，之后的（更多丰富的接口需要去完善一下）

import gym
import numpy as np
import os
import sys
import traci
import traci.constants as tc
from traci import domain
from gym import spaces


class Freewheeling_Intersection_V0(gym.Env):
    """
    Description:
        A traffic signal control simulator for an isolated intersection.
        We supposed that there is no concept of cycle in the signal control.
        Hence you may execute one specific phase again before the others are executed.
        When one particular phase is over, it's time to decide(choose action) which phase(DISCRETE)
        and the  phase's duration(CONTINUOUS).
        If you take advantage of this env, it's a RL problem with hybrid action space actually,
        but if you just want to train and evaluate with a NORMAL env, just add some confines in
        env or train.py. # TODO

    Observation:  # TODO

    Actions:
        Type: Discrete(8)
        Num   Action
        0     NS_straight
        1     EW_straight
        2     NS_left
        3     EW_left
        4     N_straight_left
        5     E_straight_left
        6     S_straight_left
        7     W_straight_left

    -------------- PLUS ----------
        Type: Box(8)
        Num   Action              Min      Max
        0     NS_straight          5        20
        1     EW_straight          5        20
        2     NS_left              5        20
        3     EW_left              5        20
        4     N_straight_left      5        20
        5     E_straight_left      5        20
        6     S_straight_left      5        20
        7     W_straight_left      5        20

    Reward:  # TODO

    Starting State:  # TODO

    Episode Termination:
        Episode length is greater than 3600.
    """

    def __init__(self):

        action_low = np.array([5.] * 8)
        action_high = np.array([20.] * 8)
        self.action_space = spaces.Tuple((
            spaces.Discrete(8),
            spaces.Tuple(
                tuple(spaces.Box(action_low[i], action_high[i], dtype=np.float32) for i in range(8))
            )
        ))
        self.observation_space = spaces.Tuple((



        ))

        # declare the path to sumo/tools
        # sys.path.append('/path/to/sumo/tools')
        sys.path.append('D:/SUMO/tools')

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.

        :return: dic, speed and position of different vehicle types
        """
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

        return vehicles_speed, vehicles_position

    def step(self, action):
        """

        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step' needs as you can see.
              Thus only a new action is input  will we change the traffic light state, otherwise just do
              traci.simulationStep()


        :param action: array [1 * 2], the former is the phase next period, and the latter is its duration respectively.
        :return: next_state, reward, done, _
        """

        assert action is np.array , 'wrong data structure for action'
        phase_next = action[0]
        phase_duration = action[1]














