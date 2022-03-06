# @author Metro
# @time 2021/11/11
# happy to be single!
# 暂时不考虑加入参数
# 先实现自己的状态空间，之后的（更多丰富的接口需要去完善一下）


import gym
import numpy as np
import sys
import random
import copy
import traci
import traci.constants as tc
from gym import spaces


class FreewheelingIntersectionEnv_v1(gym.Env):
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
        Type: Box(8)
        Num  Observation                   Min      Max
        0    Phase_0  queue                 0.       100.
                            ...
        7    Phase_7  queue                 0.       100.
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
        Type: Box(1)
        Num   Action                                           Min      Max
        0     The duration of phase you have selected          10       30
    Reward:
        Mean travel time of vehicles depart the ''input edge'' during this signal stage.
    Starting State:
        Initialization according to sumo, actually there is no vehicles at the beginning
    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(1800 in default, for half an hour).
    """

    def __init__(self):
        self.phase_num = 8

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
            [24, 12, 25, 12, None, 12, 12, 12],
            [13, 26, 13, 27, 13, None, 13, 13],
            [28, 14, 29, 14, 14, 14, None, 14],
            [15, 30, 15, 31, 15, 15, 15, None]
        ])

        self.lane_length = 240.
        self.yellow = 3
        self.max_queuing_speed = 1.
        self.simulation_steps = 1800
        self.episode_steps = 0

        self.action_pre = None
        self.vehicle_pre = None

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.phase_num),
            spaces.Box(low=np.array([10]), high=np.array([30]), dtype=np.float32)
        ))

        observation_low = np.array([0.] * self.phase_num)
        observation_high = np.array([1.] * self.phase_num)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        seed = 1
        self.seed(seed)

        # declare the path to sumo/tools
        # sys.path.append('/path/to/sumo/tools')
        sys.path.append('D:/SUMO/tools')

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.
        :return: number of vehicles w.r.t eight phases
        """

        path = 'envs/sumo/road_network/FW_Inter.sumocfg'

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        self.episode_steps = 0
        raw = self.retrieve_raw_info()
        state = self.retrieve_state(raw)

        self.action_pre = []

        return state

    def step(self, action):
        """
        :param action:list, e.g. [4, 12],
                             the first element is the stage next period, and the latter one is its duration
        :return: next_state, reward, done, info
        """

        stage_next = action[0]
        stage_duration = action[1]

        # SmartWolfie is a traffic signal control program defined in FW_Inter.add.xml.
        # We achieve hybrid action space control through switch its stage and steps
        # (controlled by YELLOW(3s in default) and GREEN(stage_duration)).
        # There is possibility that stage next period is same with the stage right now
        # Otherwise there is a yellow between two different stages.

        if len(self.action_pre) and action[0] != self.action_pre[0]:
            yellow = self.phase_transformer[self.action_pre[0]][action[0]]
            traci.trafficlight.setPhase('SmartWolfie', yellow)
            for t in range(self.yellow):
                traci.simulationStep()
                self.episode_steps += 1

        traci.trafficlight.setPhase('SmartWolfie', stage_next)
        for t in range(stage_duration):
            traci.simulationStep()
            self.episode_steps += 1

        raw = self.retrieve_raw_info()
        # state
        state = self.retrieve_state(raw)

        # reward
        vehicle_now, waiting_time = self.retrieve_reward(raw)
        departed_vehicle = list(set(self.vehicle_pre) - set(vehicle_now))
        reward = len(departed_vehicle)

        # done
        if self.episode_steps > self.simulation_steps:
            done = 1
        else:
            done = 0

        # info
        info = self.retrieve_more_info(raw)

        self.action_pre = copy.deepcopy(action)
        self.vehicle_pre = copy.deepcopy(vehicle_now)

        return state, reward, done, info

    def retrieve_raw_info(self):
        """
        :return: dic to save vehicles' speed and position etc. w.r.t its vehicle type
        """
        vehicle_raw_info = {}

        for edgeID in self.edgeIDs:
            vehicle_raw_info[edgeID] = []
            traci.edge.subscribe(edgeID, (tc.LAST_STEP_VEHICLE_ID_LIST,))
            vehicle_IDs_edge = [str(_) for _ in traci.edge.getSubscriptionResults(edgeID).values()]
            for ID in vehicle_IDs_edge:
                traci.vehicle.subscribe(ID, (tc.VAR_TYPE, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                                             tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_TIMELOSS))
                temp = [_ for _ in traci.vehicle.getSubscriptionResults(ID).values()]
                temp[1] = self.lane_length - temp[1]
                temp.insert(0, ID)
                vehicle_raw_info[edgeID].append(temp)
                # LENGTH_LANE is the length of lane, gotten from FW_Inter.net.xml.
                # temp[0]:str, vehicle's ID
                # temp[1]:float, the distance between vehicle and lane's stop line.
                # temp[2]:float, speed
                # temp[3]:float, accumulated_waiting_time
                # temp[4]:float, time loss

        return vehicle_raw_info

    def retrieve_state(self, raw):
        """
        :return:
        """

        state = np.array([len(v) for k, v in raw.items()])

        return state

    def retrieve_reward(self, raw):
        """
        :return:
        """
        vehicle_IDs = []
        acc_waiting_time = []
        for k, v in raw.items():
            vehicle_IDs_edge = [_[0] for _ in v]
            acc_waiting_time_edge = [_[3] for _ in v]
            vehicle_IDs.append(vehicle_IDs_edge)
            acc_waiting_time.append(acc_waiting_time_edge)

        return vehicle_IDs, acc_waiting_time

    def retrieve_more_info(self, raw):
        """
        Mainly for evaluation

        :param raw:
        :return:
        """
        queue = []
        raw = list(raw.items())
        for vehicles_specific_type in raw:
            for vehicle in vehicles_specific_type[1]:
                if vehicle[2] < self.max_queuing_speed:
                    queue.append(vehicle)

        return len(queue)

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


if __name__ == "__main__":
    fw = FreewheelingIntersectionEnv()
    fw.reset()
    for i in range(50):
        fw.sumo_step()
    raw_info = fw.retrieve_raw_info()
    print(raw_info)
    fw.retrieve_state(raw_info)
