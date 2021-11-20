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
from gym.utils import seeding


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

        # for every vehicle type the maximum recorded number is 25 w.r.t its position(padded with 'inf') and speed
        # (padded with '0')
        self.PAD_LENGTH = 25
        self.LENGTH_LANE_HIGH = 250
        self.SPEED_HIGH = 100
        observation_low = np.array([0.] * 8 * 2 * self.PAD_LENGTH)
        observation_high = np.concatenate((np.array([self.LENGTH_LANE_HIGH] * self.PAD_LENGTH),
                                           np.array([self.SPEED_HIGH] * self.PAD_LENGTH)), axis=1)
        observation_high = np.tile(observation_high, 8)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high
        )
        self.seed()

        # declare the path to sumo/tools
        # sys.path.append('/path/to/sumo/tools')
        sys.path.append('D:/SUMO/tools')

        # the edgeID is defined in FW_Inter.edg.xml
        # as you may have different definition in your own .edg.xml, change it.
        self.edgeIDs = ['north_in', 'east_in', 'south_in', 'west_in']

        # vehicle_types will help to filter the vehicles on the same edge but have different direction.
        self.vehicle_types = ['NW_right', 'NS_through', 'NE_left',
                              'EN_right', 'EW_through', 'ES_left',
                              'SE_right', 'SN_through', 'SW_left',
                              'WS_right', 'WE_through', 'WN_left']

        self.steps_one_episode = 0
        self.reward_last_phase = 0
        self.alpha = 0.2
        self.YELLOW = 3
        self.LENGTH_LANE = 234.12

        self.SIMULATION_STEPS = 3600
        # when step() we will save last 'self.N_STEPS' states for state representation
        self.N_STEPS = 5

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.

        :return: dic, speed and position of different vehicle types
        """
        path = '../envs/sumo/road_network/FW_Inter.sumocfg'

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        self.steps_one_episode = 0
        raw = self.retrieve()
        # info = sum(info.values(), [])
        state_one_step = self.get_state_one_step(raw)
        self.reward_last_phase = self.get_reward(raw)

        return state_one_step

    def step(self, action):
        """

        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step()' needs as you can see.
              Thus only a new action is input will we change the traffic light state, otherwise just do
              traci.simulationStep()


        :param action:array, e.g. array([4, [0, 0, 0, 0, 10, 0, 0, 0]]),
                             the former is the phase next period, and the latter is duration list w.r.t all phases.
        :return: next_state, reward, done, _
        """

        phase_next = action[0]
        phase_duration = action[1][phase_next]
        action_old = None

        state_one_step = []
        state_n_steps = []

        # SmartWolfie is a traffic light control program defined in FW_Inter.add.xml We achieve hybrid action space
        # controlling through switch its phase and steps(controlled by self.YELLOW and GREEN(phase_duration)).
        # When the phase is changed(there is possibility that phase next period is same with the phase right now),

        if not action_old:
            pass
        else:
            if action[0] == action_old[0]:
                # phase next period is same with the phase right now, just accumulate the duration
                pass
            else:
                traci.trafficlight.setPhase('SmartWolfie', action_old[1][8 + phase_next])
                for i in range(self.YELLOW):
                    traci.simulationStep()
                    self.steps_one_episode += 1
        traci.trafficlight.setPhase('SmartWolfie', phase_next)
        for i in range(phase_duration):
            traci.simulationStep()
            self.steps_one_episode += 1
            if phase_duration - i <= self.N_STEPS:
                raw = self.retrieve()
                state_one_step = self.get_state_one_step(raw)
                state_n_steps.append(state_one_step)
        reward_present_phase = self.get_reward(raw)
        reward = reward_present_phase - self.reward_last_phase
        self.reward_last_phase = copy.deepcopy(reward_present_phase)

        action_old = copy.deepcopy(action)
        if self.steps_one_episode > self.SIMULATION_STEPS:
            done = True
        else:
            done = False
        info = {}
        return state_n_steps, reward, done, info

    def retrieve(self):
        """

        :return:
        """
        # dic to save vehicles' speed and position etc. w.r.t its vehicle type
        # e.g. vehicles_speed = {'NW_right':['vehicle_id_0', position, speed, accumulated_waiting_time, time_loss],...
        #                        'NS_through':...}
        vehicles_raw_info = {}

        for edgeID in self.edgeIDs:
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
                    tem[1] = self.LENGTH_LANE - tem[1]
                    # LENGTH_LANE is the length of  lane, gotten from FW_Inter.net.xml.
                    # ID:str, vehicle's ID
                    # tem[1]:float, the distance between vehicle and lane's stop line.
                    # tem[2]:float, speed
                    # tem[3]:float, accumulated_waiting_time
                    # tem[4]:float, time loss
                    if tem[0] not in vehicles_raw_info:
                        vehicles_raw_info[tem[0]] = []
                    vehicles_raw_info[tem[0]].append([ID, tem[1], tem[2], tem[3], tem[4]])

        return vehicles_raw_info

    def get_state_one_step(self, raw):
        """

        :return:
        """
        state_one_step = []
        raw = list(raw.items())
        for vehicles_specific_type in raw:
            position_specific_type = []
            speed_specific_type = []
            for vehicle in vehicles_specific_type[1]:
                position_specific_type.append(vehicle[1])
                speed_specific_type.append(vehicle[2])
            position_specific_type = sum(position_specific_type, [])
            speed_specific_type = sum(speed_specific_type, [])
            np.pad(position_specific_type, (0, self.PAD_LENGTH - len(position_specific_type)), 'constant',
                   constant_values=(0, float('inf')))
            np.pad(speed_specific_type, (0, self.LENGTH_LANE - len(speed_specific_type)), 'constant',
                   constant_values=(0, 0))
            state_one_step.append(position_specific_type)
            state_one_step.append(speed_specific_type)
        state_one_step = sum(state_one_step, [])

        return state_one_step

    def get_reward(self, raw):
        """
        Temporarily, we just use 'queue' and 'time loss' to design the reward.
        Alpha is a trade off between the influence of 'queue' and 'time loss'.

        :return:
        """
        loss_time = []
        queue = []
        raw = list(raw.items())
        for vehicles_specific_type in raw:
            loss_time_specific_type = []
            for vehicle in vehicles_specific_type[1]:
                loss_time_specific_type.append(vehicle[4])
            loss_time.append(loss_time_specific_type)
            queue.append(len(loss_time_specific_type))
        loss_time = sum(loss_time, [])
        reward = -(np.mean(loss_time) + self.alpha * np.mean(loss_time))

        return reward

    def seed(self, seed=None):  # TODO
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        """

        :return:
        """
        traci.close()
