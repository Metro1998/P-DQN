# @author Metro
# @time 2021/11/11
# happy to be single!
# 暂时不考虑加入参数
# 先实现自己的状态空间，之后的（更多丰富的接口需要去完善一下）

import gym
import os
import numpy as np
import sys
import random
import copy
import traci
import traci.constants as tc
from gym import spaces
from bisect import bisect_left


class FreewheelingIntersectionEnv(gym.Env):
    """
    Description:
        A traffic signal control simulator environment for an isolated intersection.

        We supposed that there is no concept of cycle in the signal control.Hence you may execute one specific phase
        repeatedly before the others are executed.
        When one particular phase is over, it's time to decide(choose action) which phase(DISCRETE) to execute and its
        duration(int(CONTINUOUS)).

        It's a RL problem with hybrid action space actually, but if you just want to train and evaluate with a
        NORMAL env, just add some confines in env or train.py.

    Observation:
        Type: Box(400)
        # 400 = 8 * 25 * 2
        # 8 phases
        # We only record the first 25th vehicles(fixed number) on each lane(?)
        # The distance to the stop line and speed are considered
        # When vehicles are absent on one lane, pad it with float('inf') and 0 w.r.t position and speed.
        Num  Observation                   Min      Max
        0    Phase_0 vehicle_0 position     0       250
                            ...
        25   Phase_0 vehicle_0 speed        0       100
                            ...

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
        0     NS_straight          10       25
        1     EW_straight          10       25
        2     NS_left              10       25
        3     EW_left              10       25
        4     N_straight_left      10       25
        5     E_straight_left      10       25
        6     S_straight_left      10       25
        7     W_straight_left      10       25

    Reward:
        A combination between vehicle's loss time and queue in one specific phase.

    Starting State:
        Initialization according to sumo, actually there is no vehicles at the beginning

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for one hour).
    """

    def __init__(self):
        self.phase_num = 8

        # for every vehicle type the maximum recorded number is 25 w.r.t its position(padded with 'inf') and speed
        # (padded with '0')
        self.cells = 32

        self.lane_length_high = 250.
        self.speed_high = 100.

        # the edgeID is defined in FW_Inter.edg.xml
        # as you may have different definition in your own .edg.xml, change it in config.
        self.edgeIDs = ['north_in', 'east_in', 'south_in', 'west_in']

        # vehicle_types will help to filter the vehicles on the same edge but have different direction.
        self.vehicle_types = ['NS_through', 'NE_left',
                              'EW_through', 'ES_left',
                              'SN_through', 'SW_left',
                              'WE_through', 'WN_left']

        self.phase_transformer = np.array([
            [None, 8, 8, 8, 16, 8, 17, 8],
            [9, None, 9, 9, 9, 18, 9, 19],
            [10, 10, None, 10, 20, 10, 21, 10],
            [11, 11, 11, None, 11, 22, 11, 23],
            [21, 12, 17, 12, None, 12, 12, 12],
            [13, 23, 13, 19, 13, None, 13, 13],
            [20, 14, 16, 14, 14, 14, None, 14],
            [15, 22, 15, 18, 15, 15, 15, None]
        ])

        self.yellow = 3
        self.lane_length = 234.13
        self.max_queuing_speed = 1.
        self.simulation_steps = 3600

        # when step() we will save last 'self.N_STEPS' states for state representation
        self.n_steps = 5

        self.alpha = 0.2
        self.episode_steps = 0
        self.reward_previous = []
        self.states = []

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.phase_num),
            spaces.Box(low=np.array([10]), high=np.array([25]), dtype=np.float32)
        ))

        observation_low = np.array([0.] * self.phase_num * 2 * self.pad_length)
        observation_high = np.concatenate((np.array([self.lane_length_high] * self.pad_length),
                                           np.array([self.speed_high] * self.pad_length)), axis=None)
        observation_high = np.tile(observation_high, self.phase_num)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32
        )
        seed = 1
        self.seed(seed)

        # declare the path to sumo/tools
        # sys.path.append('/path/to/sumo/tools')
        sys.path.append('D:/SUMO/tools')

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.

        :return: dic, speed and position of different vehicle types
        """
        print(os.getcwd())
        path = 'envs/sumo/road_network/FW_Inter.sumocfg'

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        self.episode_steps = 0
        self.action_old = []
        raw_info = self.retrieve_raw_info()
        state = self.retrieve_state(raw_info)
        self.reward_previous = [0] * 4

        return np.array(state, dtype=np.float32)

    def sumo_step(self):
        """
        SUMO steps.

        :return:
        """
        traci.simulationStep()
        self.episode_steps += 1
        raw_info = self.retrieve_raw_info()
        state = self.retrieve_state(raw_info)
        self.states.append(state)

    def step(self, action):
        """

        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step()' needs as you can see.
              Thus only a new action is input will we change the traffic light state, otherwise just do
              traci.simulationStep() consecutively.


        :param action:list, e.g. [4, 12, 11, 13, 15, 10, 12, 16, 23],
                             the first element is the phase next period,
                             and the latter ones are duration w.r.t all phases.
        :return: next_state, reward, done, info
        """

        phase_next = action[0]
        phase_duration = action[1]
        self.states = []

        # SmartWolfie is a traffic light control program defined in FW_Inter.add.xml We achieve hybrid action space
        # control through switch its phase and steps(controlled by self.YELLOW and GREEN(phase_duration)).
        # There is possibility that phase next period is same with the phase right now

        if not self.action_old:
            pass
        else:
            if action[0] == self.action_old[0]:
                # Phase next period is same with the phase right now, just accumulate the duration
                pass
            else:
                # When phase is changed, YELLOW PHASE is executed.
                yellow_phase = self.phase_transformer[self.action_old[0]][action[0]]
                traci.trafficlight.setPhase('SmartWolfie', yellow_phase)
                for i in range(self.yellow):
                    self.sumo_step()

        traci.trafficlight.setPhase('SmartWolfie', phase_next)
        for i in range(int(np.ceil(phase_duration))):
            self.sumo_step()

        # ---- states ----
        state = np.array(self.states[-1], dtype=float)
        print(state)

        # ---- reward ----
        raw_info = self.retrieve_raw_info()
        reward_so_far = self.retrieve_reward(raw_info)
        reward = reward_so_far[0] - self.reward_previous[0]

        self.reward_previous = copy.deepcopy(reward_so_far)
        self.action_old = copy.deepcopy(action)

        if self.episode_steps > self.simulation_steps:
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, info

    def retrieve_raw_info(self):
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
                    tem[1] = self.lane_length - tem[1]
                    # LENGTH_LANE is the length of  lane, gotten from FW_Inter.net.xml.
                    # ID:str, vehicle's ID
                    # tem[1]:float, the distance between vehicle and lane's stop line.
                    # tem[2]:float, speed
                    tem[2] *= 3.6
                    # tem[3]:float, accumulated_waiting_time
                    # tem[4]:float, time loss
                    if tem[0] not in vehicles_raw_info:
                        vehicles_raw_info[tem[0]] = []
                    vehicles_raw_info[tem[0]].append([ID, tem[1], tem[2], tem[3], tem[4]])

        return vehicles_raw_info

    def retrieve_state(self, raw_info):
        """

        :return:
        """

        vehicle_types_so_far = []
        state = np.array([])
        cell_space = np.linspace(0, 240, num=(self.cells + 1))

        raw = list(raw_info.items())
        for type in raw:
            vehicle_types_so_far.append(type[0])
        for vehicle_type in self.vehicle_types:
            position = np.zeros(self.cells)
            speed = np.zeros(self.cells)
            if vehicle_type in vehicle_types_so_far:
                for vehicle in raw_info[vehicle_type]:
                    position[bisect_left(cell_space, vehicle[1])]
                    speed[bisect_left(cell_space), vehicle[1]]
            state = np.concatenate((state, position), axis=0)
            state = np.concatenate((state, speed), axis=0)
        state = state.flatten()

        return state

    def retrieve_reward(self, raw):
        """

        :return:
        """
        loss_time = []
        accumulated_waiting_time = []
        speed = []
        queue = []
        reward = np.array([0.] * 3)

        raw = list(raw.items())
        for vehicles_specific_type in raw:
            speed_specific_type = []
            accumulated_waiting_time_specific_type = []
            loss_time_specific_type = []
            for vehicle in vehicles_specific_type[1]:
                speed_specific_type.append(vehicle[2])
                accumulated_waiting_time_specific_type.append(vehicle[3])
                loss_time_specific_type.append(vehicle[4])
            for speed in speed_specific_type:
                if speed < self.max_queuing_speed:
                    queue.append(len(speed_specific_type[-speed_specific_type.index(speed):]))
                    break
            accumulated_waiting_time.append(accumulated_waiting_time_specific_type)
            loss_time.append(loss_time_specific_type)
        loss_time = sum(loss_time, [])
        accumulated_waiting_time = sum(accumulated_waiting_time, [])

        reward[0] = np.sum(queue)  # total queue at present
        reward[1] = np.mean(loss_time)  # average loss time
        reward[2] = np.mean(accumulated_waiting_time)  # average waiting time

        return reward

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        """

        :return:
        """
        traci.close()
