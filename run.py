# @author Metro
# @time 2021/11/24
import click
import gym
from agents.pdqn import PDQNBaseAgent
from utilities.memory.memory import ReplayBuffer


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        self.env = gym.make('FreewheelingIntersection_v0')

        # Agent
        self.agent = PDQNBaseAgent(config)

        # Memory
        self.replay_memory = ReplayBuffer()



