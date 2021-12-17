# @author Metro
# @time 2021/11/24
import os.path

import gym
from agents.pdqn import PDQNBaseAgent
from utilities.memory import ReplayBuffer
from utilities.utilities import *
from utilities.route_generator import generate_routefile


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        generate_routefile(seed=config.seed, demand=config.demand)
        self.env = gym.make(config.environment)

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
        global i_episode
        rolling_scores_for_diff_runs = []
        file_to_save_actor = os.path.join(self.file_to_save, 'actor/')
        file_to_save_actor_param = os.path.join(self.file_to_save, 'actor_param/')
        file_to_save_runs = os.path.join(self.file_to_save, 'runs/')
        file_to_save_rolling_scores = os.path.join(self.file_to_save, 'rolling_scores/')
        os.makedirs(file_to_save_actor, exist_ok=True)
        os.makedirs(file_to_save_actor_param, exist_ok=True)
        os.makedirs(file_to_save_runs, exist_ok=True)
        os.makedirs(file_to_save_rolling_scores, exist_ok=True)

        for run in range(self.runs_per_agent):
            game_full_episodes_scores = []
            game_full_episodes_rolling_scores = []

            for i_episode in range(self.maximum_episodes):

                if self.save_freq > 0 and i_episode % self.save_freq == 0:
                    actor_path = os.path.join(file_to_save_actor, 'episode{}'.format(i_episode))
                    actor_param_path = os.path.join(file_to_save_actor_param, 'episode{}'.format(i_episode))
                    self.agent.save_models(actor_path, actor_param_path)

                # We temporarily regard reward and scores as the same thing
                # TODO
                episode_reward = 0
                episode_steps = 0
                done = False
                state = self.env.reset()  # n_steps

                while not done:
                    if self.total_steps < self.randomly_pick_steps:
                        action, all_action_param = self.agent.randomly_pick()
                    else:
                        action, action_param, all_action_param = self.agent.pick_action(state, self.train)

                    action_for_env = tuple([action, all_action_param[action]])
                    print(action_for_env)
                    if len(self.memory) > self.batch_size:
                        for i in range(self.updates_per_step):
                            self.agent.optimize_td_loss(self.memory)
                            self.total_updates += 1

                    next_state, reward, done, _ = self.env.step(action_for_env)

                    episode_steps += 1
                    print(episode_reward)
                    episode_reward += reward

                    self.total_steps += 1

                    self.memory.push(state, action, all_action_param, reward, next_state, done)

                    state = next_state

                episode_score_so_far = episode_reward
                game_full_episodes_scores.append(episode_score_so_far)
                game_full_episodes_rolling_scores.append(
                    np.mean(game_full_episodes_scores[-1 * self.rolling_score_window:]))

                print("Episode: {}, total steps:{}, episode steps:{}, scores:{}".format(
                    i_episode, self.total_steps, episode_steps, episode_score_so_far))

                self.env.close()
                file_path_for_pic = os.path.join(file_to_save_runs, 'episode{}_run{}.jpg'.format(i_episode, run))
                visualize_results_per_run(agent_results=game_full_episodes_scores,
                                          agent_name=self.agent_name,
                                          save_freq=1,
                                          file_path_for_pic=file_path_for_pic)
                rolling_scores_for_diff_runs.append(game_full_episodes_rolling_scores)

        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'rolling_scores.jpg')
        visualize_overall_agent_results(agent_results=rolling_scores_for_diff_runs,
                                        agent_name=self.agent_name,
                                        show_mean_and_std_range=True,
                                        agent_to_color_dictionary=self.agent_to_color_dictionary,
                                        standard_deviation_results=1,
                                        file_path_for_pic=file_path_for_pic
                                        )
