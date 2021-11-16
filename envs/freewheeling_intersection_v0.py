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

        :return:
        """
        # sumoBinary = "/path/to/sumo"
        sumoBinary = 'D:/SUMO/bin/sumo'
        sumoCmd = [sumoBinary, '-c', 'FW_Inter.sumocfg']

        # create instances
        traci.start(sumoCmd, label='sim1')
        traci.domain.






