# @author Metro
# @time 2021/11/24
import click
import gym
from agents.pdqn import PDQNAgent
from utilities.memory.memory import Memory


class Train_and_Evaluate(object):

    def __init__(self, config):
        self.capacity = config.hyperparameters[replay_memory_size]
        self.env = gym.make('FreewheelingIntersection_v0')
        self.agent = PDQNAgent(config)
        self.replay_memory = Memory(limit=self.capacity, observation_shape=self.env.observation_space.shape,
                                    action_shpae=(1 + config.env_parameters[PHASE_NUM],), next_actions=False)



