# @author Metro
# @time 2021/10/29

import os
import random
import numpy as np
import torch
import gym
from torch.nn.utils import clip_grad_norm_


class Base_Agent(object):
    """
    Define a basic reinforcement learning agent
    """

    NAME = "Abstract Agent"

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.set_random_seeds(self.seed)
        self.num_episodes_to_run = config.num_episodes_to_run
        self.environment = gym.make(config.environment)
        self.env_parameters = config.env_parameters
        self.hyperparameters = config.hyperparameters
        self.use_GPU = config.use_GPU
        self.device = torch.device(self.hyperparameters['device'])

        self.file_to_save = config.file_to_save
        self.runs_per_agent = config.runs_per_agent
        self.randomise_random_seed = config.randomise_random_seed

    def set_random_seeds(self, random_seed):
        """
        Sets all possible random seeds to results can be reproduces.

        :param random_seed:
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def pick_action(self, state):
        """
        Determines which action to take when given state

        :param state:
        :return:
        """
        raise NotImplementedError

    def start_episode(self):
        """
        Perform any initialisation for the start of an episode.
        :return:
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Performs any cleanup before the next episode.
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc
