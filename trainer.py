# @author Metro
# @time 2021/11/24

import time
import gym
import matplotlib.pyplot as plt
import numpy as np
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

        self.train = config.train
        self.evaluate = config.evaluate
        self.evaluate_internal = config.evaluate_internal

        self.agent_to_color_group = config.agent_to_color_dictionary
        self.standard_deviation_results = config.standard_deviation_results

        # Training Loop

    def train(self):
        """

        :return:
        """
        start_time = time.time()

        for i_episode in range(self.maximum_episodes):

            if self.save_freq > 0 and self.file_to_save_actor and self.file_to_save_actor_param \
                    and i_episode % self.save_freq == 0:
                self.agent.sava_model(self.file_to_save_actor, self.file_to_save_actor_param)

            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()  # n_steps
            state = state[:-1]  # The last piece of n_steps

            while not done:
                if self.total_steps < self.randomly_pick_steps:
                    action = self.agent.randomly_pick()
                else:
                    action = self.agent.pick_action(state, self.train)

                if len(self.memory) > self.batch_size:
                    for i in range(self.updates_per_step):
                        self.agent.optimize_td_loss(self.memory)
                        self.total_updates += 1

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state[:-1]
                reward = reward[0]
                episode_steps += 1
                episode_reward += reward
                self.total_steps += 1

                self.memory.push(state, action, reward, next_state, done)

                state = next_state

            print("Episode: {}, total steps:{}, episode steps:{}, reward:{}".format(
                i_episode, self.total_steps, episode_steps, episode_reward))

    def visualize_overall_agent_results(self, agent_results, agent_name, show_mean_and_std_range=True,
                                        show_each_run=False, color=None, ax=None, title=None, y_limits=None):
        """
        Visualize the results for one agent.

        :param agent_results:
        :param agent_name:
        :param show_mean_and_std_range:
        :param show_each_run:
        :param color:
        :param ax:
        :param title:
        :param y_limits:
        :return:
        """
        assert isinstance(agent_results, list), 'agent_results must be a list of lists.'
        assert isinstance(agent_results[0], list), 'agent_result must be a list of lists.'
        if not ax:
            ax = plt.gca()
        if not color:
            color = self.agent_to_color_group[agent_name]
        if show_mean_and_std_range:
            mean_minus_x_std, mean_results, mean_plus_x_std = self.get_mean_and_standard_deviation_difference(agent_results)
            x_vals = list(range(len(mean_results)))
            ax.plot(x_vals, mean_results, label=agent_name, color=color)
            ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
            ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
            ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
        else:
            for ix, result in enumerate(agent_results):
                x_vals = list(range(len(agent_results[0])))
                ax.plot(x_vals, requests, label=agent_name + '_{}'.format(ix + 1), color=color)
                # Necessity to change the color


    def get_mean_and_standard_deviation_difference(self, results):
        """
        From a list of lists of specific agent results it extracts the mean result and the mean result plus or minus
        some multiple of standard deviation.

        :param results:
        :return:
        """
        def get_results_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return results_at_a_time_step
        def get_std_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)
        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
        mean_minus_x_std = [mean_val - self.standard_deviation_results * get_std_at_a_time_step(results, timestep)
                            for timestep, mean_val in enumerate(mean_results)]
        mean_plus_x_std = [mean_val - self.standard_deviation_results * get_std_at_a_time_step(results, timestep)
                            for timestep, mean_val in enumerate(mean_results)]
        return mean_minus_x_std, mean_results, mean_plus_x_std





