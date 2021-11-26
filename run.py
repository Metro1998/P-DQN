# @author Metro
# @time 2021/11/24

import time
import click
import gym
import itertools
from agents.pdqn import PDQNBaseAgent
from utilities.memory.memory import ReplayBuffer


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        self.env = gym.make('FreewheelingIntersection_v0')

        # Agent
        self.agent = PDQNBaseAgent(config)

        # Memory
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.batch_size = config.hyperparameters['batch_size']
        self.updates_per_step = config.hyperparameters['updates_per_step']
        self.memory = ReplayBuffer(self.replay_memory_size)

        self.total_steps = 0
        self.total_updates = 0
        self.randomly_pick_steps = config.hyperparameters['random_pick_steps']

        self.save_freq = config.save_freq
        self.file_to_save_actor = config.file_to_save_actor
        self.file_to_save_actor_param = config.file_to_save_actor_param

        self.maximum_episodes = config.hyperparameters['maximum_episodes']

        # Training Loop
    def train_and_evaluate(self):
        """

        :return:
        """
        start_time = time.time()
        train = True

        for i_episode in range(self.maximum_episodes):

            if self.save_freq > 0 and self.file_to_save_actor and self.file_to_save_actor_param \
                    and i_episode % self.save_freq == 0:
                self.agent.sava_model(self.file_to_save_actor, self.file_to_save_actor_param)

            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()

            while not done:
                if self.total_steps < self.randomly_pick_steps:
                    action = self.agent.randomly_pick()
                else:
                    action = self.agent.pick_action(state, train)

                if len(self.memory) > self.batch_size:
                    for i in range(self.updates_per_step):
                        self.agent.optimize_td_loss(self.memory)
                        self.total_updates += 1

                next_state, reward, done, _ = self.env.step(action)
                episode_steps += 1
                episode_reward += reward
                self.total_steps += 1

                self.memory.push(state, action, reward, next_state, done)

                state = next_state







