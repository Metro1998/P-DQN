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


class FreewheelingIntersectionEnv(gym.Env):
    """
    Description:
        A traffic signal control simulator environment for an isolated intersection.

        We supposed that there is no concept of cycle in the signal control.Hence you may execute one specific phase
        repeatedly before the others are executed.
        When one particular phase is over, it's time to decide(choose action) which phase(DISCRETE) to execute and its
        duration(CONTINUOUS).

        It's a RL problem with hybrid action space actually, but if you just want to train and evaluate with a
        NORMAL env, just add some confines in env or train.py.

    Observation:
        Type: Box(400)
        # 400 = 8 * 25 * 2
        # 8 phases
        # We only record the first 25th vehicles in each phase
        # The distance to the stop line and speed are considered
        # When vehicles are absent in one phase, pad it with float('inf') and 0 w.r.t position and speed.
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
        0     NS_straight          5        20
        1     EW_straight          5        20
        2     NS_left              5        20
        3     EW_left              5        20
        4     N_straight_left      5        20
        5     E_straight_left      5        20
        6     S_straight_left      5        20
        7     W_straight_left      5        20

    Reward:
        A combination between vehicle's loss time and queue in one specific phase.

    Starting State:
        Initialization according to sumo, actually there is no vehicles at the beginning

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for one hour).
    """

    def __init__(self, config):
        self.env_parameters = config.env_parameters
        self.phase_num = self.env_parameters['phase_num']
        self.action_low = self.env_parameters['action_low']
        self.action_high = self.env_parameters['action_high']

        # for every vehicle type the maximum recorded number is 25 w.r.t its position(padded with 'inf') and speed
        # (padded with '0')
        self.pad_length = self.env_parameters['pad_length']

        self.lane_length_high = self.env_parameters['lane_length_high']
        self.speed_high = self.env_parameters['speed_high']

        # the edgeID is defined in FW_Inter.edg.xml
        # as you may have different definition in your own .edg.xml, change it in config.
        self.edgeIDs = self.env_parameters['edge_ids']

        # vehicle_types will help to filter the vehicles on the same edge but have different direction.
        self.vehicle_types = self.env_parameters['vehicle_types']

        self.yellow = config.env_parameters['yellow']
        self.lane_length = config.env_parameters['lane_length']
        self.simulation_steps = config.env_parameters['simulation_steps']

        # when step() we will save last 'self.N_STEPS' states for state representation
        self.n_steps = config.env_parameters['n_steps']

        self.alpha = config.env_parameters['alpha']
        self.episode_steps = 0
        self.reward_last_phase = []
        self.action_old = []

        action_low = np.array([self.action_low] * self.phase_num)
        action_high = np.array([self.action_high] * self.phase_num)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.phase_num),
            spaces.Tuple(
                tuple(spaces.Box(action_low[i], action_high[i], dtype=np.float32) for i in range(self.phase_num))
            )
        ))

        observation_low = np.array([0.] * self.phase_num * 2 * self.pad_length)
        observation_high = np.concatenate((np.array([self.lane_length_high] * self.pad_length),
                                           np.array([self.speed_high] * self.pad_length)), axis=1)
        observation_high = np.tile(observation_high, self.phase_num)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high
        )
        seed = config.seed
        self.seed(seed)

        # declare the path to sumo/tools
        # sys.path.append('/path/to/sumo/tools')
        sys.path.append('D:/SUMO/tools')

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.

        :return: dic, speed and position of different vehicle types
        """
        path = '../envs/sumo/road_network/FW_Inter.sumocfg'

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        self.episode_steps = 0
        raw = self.retrieve()
        state_one_step = self.compute_state_one_step(raw)
        self.reward_last_phase = self.compute_reward(raw)

        return np.array(state_one_step)

    def step(self, action: list):
        """

        Note: the sumo(or traci) doesn't need an action every step until one specific phase is over,
              but the abstract method 'step()' needs as you can see.
              Thus only a new action is input will we change the traffic light state, otherwise just do
              traci.simulationStep()


        :param action:array, e.g. array([4, 5, 6, 8, 6, 10, 12, 16, 9]),
                             the first element is the phase next period,
                             and the latter ones are duration list w.r.t all phases.
        :return: next_state, reward, done, _
        """

        phase_next = action[0]
        phase_duration = action[phase_next + 1]
        reward = []

        # SmartWolfie is a traffic light control program defined in FW_Inter.add.xml We achieve hybrid action space
        # control through switch its phase and steps(controlled by self.YELLOW and GREEN(phase_duration)).
        # There is possibility that phase next period is same with the phase right now

        states = []
        if not self.action_old:
            pass
        else:
            if action[0] == self.action_old[0]:
                # Phase next period is same with the phase right now, just accumulate the duration
                pass
            else:
                # When phase is changed, YELLOW PHASE is executed.
                traci.trafficlight.setPhase('SmartWolfie', action_old[1 + phase_next])
                for i in range(self.yellow):
                    traci.simulationStep()
                    self.episode_steps += 1
                    raw = self.retrieve()
                    state_one_step = self.compute_state_one_step(raw)
                    states.append(state_one_step)

        traci.trafficlight.setPhase('SmartWolfie', phase_next)
        for i in range(phase_duration):
            traci.simulationStep()
            self.episode_steps += 1
            raw = self.retrieve()
            state_one_step = self.compute_state_one_step(raw)
            states.append(state_one_step)

        states = np.array(states, dtype=float)

        raw = self.retrieve()
        reward_present_phase = self.compute_reward(raw)

        reward.append(reward_present_phase[0] - self.reward_last_phase[0])
        for i in range(3):
            reward.append(reward_present_phase[i + 1])

        reward = np.array(reward, dtype=float)
        self.reward_last_phase = copy.deepcopy(reward_present_phase)

        self.action_old = copy.deepcopy(action)
        if self.episode_steps > self.simulation_steps:
            done = True
        else:
            done = False
        info = {}
        return states, reward, done, info

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
                    tem = [None] * 5
                    traci.vehicle.subscribe(ID, (tc.VAR_TYPE, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                                                 tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_TIMELOSS))
                    for v in traci.vehicle.getSubscriptionResults(ID).values():
                        tem.append(v)
                    tem[1] = self.lane_length - tem[1]
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

    def compute_state_one_step(self, raw):
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
            np.pad(position_specific_type, (0, self.pad_length - len(position_specific_type)), 'constant',
                   constant_values=(0, float('inf')))
            np.pad(speed_specific_type, (0, self.lane_length - len(speed_specific_type)), 'constant',
                   constant_values=(0, 0))
            state_one_step.append(position_specific_type)
            state_one_step.append(speed_specific_type)
        state_one_step = sum(state_one_step, [])

        return state_one_step

    def compute_reward(self, raw):
        """
        Temporarily, we just use 'queue' and 'time loss' to design the reward.
        Alpha is a trade off between the influence of 'queue' and 'time loss'.

        :return:
        """
        loss_time = []
        accumulated_waiting_time = []
        queue = []
        reward = [0] * 4
        raw = list(raw.items())
        for vehicles_specific_type in raw:
            loss_time_specific_type = []
            accumulated_waiting_time_specific_type = []
            for vehicle in vehicles_specific_type[1]:
                loss_time_specific_type.append(vehicle[4])
                accumulated_waiting_time_specific_type.append(vehicle[3])
            loss_time.append(loss_time_specific_type)
            accumulated_waiting_time.append(accumulated_waiting_time_specific_type)
            queue.append(len(loss_time_specific_type))
        loss_time = sum(loss_time, [])
        accumulated_waiting_time = sum(accumulated_waiting_time, [])
        reward[0] = (np.mean(queue) + self.alpha * np.mean(loss_time))
        reward[1] = np.mean(queue)
        reward[2] = np.mean(loss_time)
        reward[3] = np.mean(accumulated_waiting_time)

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
