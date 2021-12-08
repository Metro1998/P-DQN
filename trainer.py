# @author Metro
# @time 2021/11/24

import gym
from agents.pdqn import PDQNBaseAgent
from utilities.memory import ReplayBuffer
from utilities.utilities import *


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        self.env_name = config.environment
        self.env = gym.make(self.env_name)

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

        self.agent_to_color_dictionary = config.agent_to_color_dictionary
        self.standard_deviation_results = config.standard_deviation_results

        self.colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        self.color_idx = 0

        self.rolling_score_window = config.rolling_score_window
        self.runs_per_agent = config.runs_per_agent
        self.agent_name = config.agent_name

        # Training Loop

    def train(self):
        """

        :return:
        """
        rolling_scores_for_diff_runs = []

        for run in range(self.runs_per_agent):
            game_full_episodes_scores = []
            game_full_episodes_rolling_scores = []

            for i_episode in range(self.maximum_episodes):

                if self.save_freq > 0 and self.file_to_save_actor and self.file_to_save_actor_param \
                        and i_episode % self.save_freq == 0:
                    self.agent.sava_model(self.file_to_save_actor, self.file_to_save_actor_param)

                # We temporarily regard reward and scores as the same thing
                # TODO
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

                total_episode_score_so_far = episode_reward
                game_full_episodes_scores.append(total_episode_score_so_far)
                game_full_episodes_rolling_scores.append(
                    np.mean(game_full_episodes_scores[-1 * self.rolling_score_window]))

                print("Episode: {}, total steps:{}, episode steps:{}, scores:{}".format(
                    i_episode, self.total_steps, episode_steps, total_episode_score_so_far))

            rolling_scores_for_diff_runs.append(game_full_episodes_rolling_scores)

        visualize_overall_agent_results(agent_results=rolling_scores_for_diff_runs,
                                        agent_name=self.agent_name,
                                        show_mean_and_std_range=True,
                                        agent_to_color_dictionary=self.agent_to_color_dictionary,
                                        standard_deviation_results=1,
                                        title='Training Result')


if __name__ == "__main__":
