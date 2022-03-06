# @author Metro
# @time 2021/11/24
import os.path

import gym
from agents.pdqn import P_DQN
from utilities.memory import ReplayBuffer
from utilities.utilities import *
from utilities.route_generator import generate_routefile


class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        generate_routefile(seed=config.seed, demand=config.demand)
        self.env = gym.make(config.environment)

        # Agent
        self.agent = P_DQN(config, self.env)

        # Memory
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.batch_size = config.hyperparameters['batch_size']
        self.memory_st = ReplayBuffer(self.replay_memory_size / 2)
        self.memory_le = ReplayBuffer(self.replay_memory_size / 2)
        self.memory_sl = ReplayBuffer(self.replay_memory_size)

        self.total_steps = 0
        self.maximum_episodes = config.maximum_episodes
        self.train = config.train
        self.updates_per_step = config.updates_per_step
        self.evaluate = config.evaluate
        self.evaluate_internal = config.evaluate_internal

        self.save_freq = config.save_freq
        self.file_to_save = config.file_to_save
        self.agent_to_color_dictionary = config.agent_to_color_dictionary

        self.rolling_score_window = config.rolling_score_window
        self.runs_per_agent = config.runs_per_agent
        self.agent_name = config.agent_name
        self.ceil = config.ceil

        # Training Loop

    def train_agent(self):
        """

        :return:
        """

        rolling_scores_for_diff_runs = []
        file_to_save_actor = os.path.join(self.file_to_save, 'actor/')
        file_to_save_actor_param = os.path.join(self.file_to_save, 'actor_param/')
        file_to_save_runs = os.path.join(self.file_to_save, 'runs_1/')
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

                episode_score = []
                state, done = self.env.reset()

                while not done:
                    min_memory_size = min(len(self.memory_st), len(self.memory_le), len(self.memory_sl))
                    if min_memory_size > self.batch_size:
                        action, action_params = self.agent.select_action(state, self.train)

                        if self.ceil:
                            action_params = np.ceil(action_params)
                        action_for_env = [action, int(action_params[action])]

                        for i in range(self.updates_per_step):
                            self.agent.update(self.memory_st, batch_size=int(self.batch_size / 4), actor_name='actor_st')
                            self.agent.update(self.memory_le, batch_size=int(self.batch_size / 4), actor_name='actor_le')
                            self.agent.update(self.memory_sl, batch_size=int(self.batch_size / 2), actor_name='actor_sl')
                    else:
                        # initial random pick
                        action_params = np.random.randint(low=10, high=31, size=8)
                        action = np.random.randint(7, size=1)[0]
                        action_for_env = [action, action_params[action]]

                    next_state, reward, done, info = self.env.step(action_for_env)

                    episode_score.append(info)

                    self.total_steps += 1

                    # push experiences into different memories
                    if action == 0 or action == 1:
                        print('s', state)
                        print('a', action)
                        print('a_p', action_params)
                        print('r', reward)
                        print('n_s', next_state)
                        print('d', done)
                        self.memory_st.push(state, action, action_params, reward, next_state, done)
                    elif action == 2 or action == 3:
                        self.memory_le.push(state, action, action_params, reward, next_state, done)
                    else:
                        self.memory_sl.push(state, action, action_params, reward, next_state, done)

                    state = next_state
                    action_pre = action

                episode_score_so_far = np.mean(episode_score)
                game_full_episodes_scores.append(episode_score_so_far)
                game_full_episodes_rolling_scores.append(
                    np.mean(game_full_episodes_scores[-1 * self.rolling_score_window:]))

                print("Episode: {}, total steps:{}, scores:{}".format(
                    i_episode, self.total_steps, episode_score_so_far))

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
