# @author Metro
# @time 2021/11/24
import os.path

import gym
import numpy as np
from agents.pdqn import P_DQN
from agents.memory.memory import ReplayBuffer
from agents.utils.route_generator import generate_routefile
from agents.utils.utilities import *


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        generate_routefile(seed=config.seed, demand=config.demand, simulation_step=config.simulation_step)
        self.env = gym.make(config.environment)

        # Agent
        self.agent = P_DQN(config, self.env)

        # Memory
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.batch_size = config.hyperparameters['batch_size']
        self.updates_per_step = config.hyperparameters['updates_per_step']
        self.memory = ReplayBuffer(self.replay_memory_size)

        self.total_steps = 0
        self.total_updates = 0

        self.save_freq = config.save_freq
        self.file_to_save = config.file_to_save
        self.maximum_episodes = config.hyperparameters['maximum_episodes']

        self.train = config.train
        self.evaluate = config.evaluate
        self.evaluate_internal = config.evaluate_internal

        self.agent_to_color_dictionary = config.agent_to_color_dictionary
        self.standard_deviation_results = config.standard_deviation_results

        self.colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        self.color_idx = 0

        self.rolling_score_window = config.rolling_score_window
        self.runs_per_agent = config.runs_per_agent
        self.agent_name = config.agent_name

        # Training Loop

    def train_agent(self):
        """

        :return:
        """

        rolling_scores_for_diff_runs = []
        file_to_save_critic = os.path.join(self.file_to_save, 'critic/')
        file_to_save_actor = os.path.join(self.file_to_save, 'actor/')
        file_to_save_runs = os.path.join(self.file_to_save, 'runs/')
        file_to_save_rolling_scores = os.path.join(self.file_to_save, 'rolling_scores/')
        os.makedirs(file_to_save_critic, exist_ok=True)
        os.makedirs(file_to_save_actor, exist_ok=True)
        os.makedirs(file_to_save_runs, exist_ok=True)
        os.makedirs(file_to_save_rolling_scores, exist_ok=True)

        for run in range(self.runs_per_agent):
            episodes_score = []
            episodes_rolling_score = []

            for i_episode in range(self.maximum_episodes):

                if self.save_freq > 0 and i_episode % self.save_freq == 0:
                    critic_path = os.path.join(file_to_save_critic, 'episode{}'.format(i_episode))
                    actor_path = os.path.join(file_to_save_actor, 'episode{}'.format(i_episode))
                    self.agent.save_models(critic_path, actor_path)

                # Initialization
                episode_score = []
                episode_steps = 0
                done = 0
                state = self.env.reset()

                while not done:
                    if len(self.memory) > self.batch_size:
                        action, action_param, action_params = self.agent.act(state)
                        action_params = np.ceil(action_params).squeeze(0).astype(np.int64)

                        action_env = np.concatenate((np.array([action]), action_params), 0)

                        for i in range(self.updates_per_step):
                            self.agent.optimize_td_loss(self.memory)
                            self.total_updates += 1
                    else:
                        action_params = np.random.randint(low=10, high=31, size=8)
                        action = np.random.randint(7, size=1)[0]
                        action_env = np.concatenate((np.array([action]), action_params), 0)

                    next_state, reward, done, info = self.env.step(action_env)

                    episode_steps += 1
                    episode_score.append(info)

                    self.total_steps += 1
                    self.memory.push(state, action, action_params, reward, next_state, done)

                    state = next_state

                episodes_score.append(np.mean(episode_score))
                episodes_rolling_score.append(
                    np.mean(episodes_score[-1 * self.rolling_score_window:]))

                self.env.close()
                self.agent.end_episode()
                file_path_for_pic = os.path.join(file_to_save_runs, 'episode{}_run{}.jpg'.format(i_episode, run))
                visualize_results_per_run(agent_results=episodes_score,
                                          agent_name=self.agent_name,
                                          save_freq=1,
                                          file_path_for_pic=file_path_for_pic)
                rolling_scores_for_diff_runs.append(episodes_rolling_score)

        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'rolling_scores.jpg')
        visualize_overall_agent_results(agent_results=rolling_scores_for_diff_runs,
                                        agent_name=self.agent_name,
                                        show_mean_and_std_range=True,
                                        agent_to_color_dictionary=self.agent_to_color_dictionary,
                                        standard_deviation_results=1,
                                        file_path_for_pic=file_path_for_pic
                                        )
