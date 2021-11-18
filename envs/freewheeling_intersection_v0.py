# @author Metro
# @time 2021/11/11
# happy to be single!
# 暂时不考虑加入参数
# 先实现自己的状态空间，之后的（更多丰富的接口需要去完善一下）

import gym
import numpy as np
import os
import sys
import copy
import traci
import traci.constants as tc
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

        self.last_time_yellow = 3
        # when step() we will save last 'self.n_steps' states
        self.n_steps = 5

        # the edgeID is defined in FW_Inter.edg.xml
        # as you may have different definition in your own .edg.xml, change it.
        self.edgeIDs = ['north_in', 'east_in', 'south_in', 'west_in']

        # vehicle_types will help to filter the vehicles on the same edge but have different direction.
        self.vehicle_types = ['NW_right', 'NS_through', 'NE_left',
                              'EN_right', 'EW_through', 'ES_left',
                              'SE_right', 'SN_through', 'SW_left',
                              'WS_right', 'WE_through', 'WN_left']

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.

        :return: dic, speed and position of different vehicle types
        """
        path = './sumo/road_network/FW_Inter.sumocfg'

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        vehicles_speed, vehicles_position = self.get_state()

        return vehicles_speed, vehicles_position

    def step(self, action):
        """

        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step()' needs as you can see.
              Thus only a new action is input will we change the traffic light state, otherwise just do
              traci.simulationStep()


        :param action: array [1 * 2], the former is the phase next period, and the latter is its duration respectively.
        :return: next_state, reward, done, _
        """

        phase_next = action[0]
        phase_duration = action[1]
        action_old = None

        vehicles_speed_n_steps = {}
        vehicles_position_n_steps = {}

        # SmartWolfie is a traffic light control program defined in FW_Inter.add.xml
        # We achieve hybrid action space control through switch its phase and steps(controlled by self.last_time_yellow
        # or self.last_time_green)
        # When the phase is changed(there is possibility that phase next period is same with the phase right now),

        if not action_old:
            pass
        else:
            if action[0] == action_old[0]:
                # phase next period is same with the phase right now, just accumulate the duration
                pass
            else:
                traci.trafficlight.setPhase('SmartWolfie', 2 * (action_old[0] - 1) + 1)
                for i in range(self.last_time_yellow):
                    traci.simulationStep()
        traci.trafficlight.setPhase('SmartWolfie', 2 * (phase_next - 1))
        for i in range(phase_duration):
            traci.simulationStep()
            if phase_duration - i <= self.n_steps:
                vehicles_speed, vehicles_position = self.get_state()
                for k, v in vehicles_speed.items():
                    if k in vehicles_speed_n_steps:
                        vehicles_speed_n_steps[k]
        action_old = copy.deepcopy(action)

    def get_state(self):
        """

        :return:
        """
        # dic to save vehicles' speed and position info w.r.t its vehicle type
        # e.g. vehicles_speed = {'NW_right':'vehicle_id_0', 'vehicle_id_6',
        #                        'NS_through':...}
        vehicles_speed = {}
        vehicles_position = {}

        # a cursor to indicate which vehicle type is being selected
        type_cursor = 0

        for edgeID in self.edgeIDs:
            traci.edge.subscribeContext(edgeID, tc.CMD_GET_VEHICLETYPE_VARIABLE,
                                        tc.LAST_STEP_VEHICLE_ID_LIST)
            vehicles_on_specific_edge = traci.edge.getContextSubscriptionResults(edgeID)
            for i in range(3):  # on one specific edge, there are three different vehicle types.
                for vehicleID in vehicles_on_specific_edge:
                    traci.vehicle.subscribeContext(vehicleID, tc.CMD_GET_VEHICLETYPE_VARIABLE,
                                                   [tc.VAR_SPEED, tc.VAR_POSITION])
                    traci.vehicle.addSubscriptionFilterVType(self.vehicle_types[type_cursor])
                    vehicles_speed[self.vehicle_types[type_cursor]] = \
                        traci.vehicle.getContextSubscriptionResults(vehicleID)
                    vehicles_position[self.vehicle_types[type_cursor]] = \
                        traci.vehicle.getContextSubscriptionResults(vehicleID)
                type_cursor += 1

        return vehicles_speed, vehicles_position
