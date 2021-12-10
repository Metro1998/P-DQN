# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import math
import torch.optim as optim
import random
from agents.base_agent import Base_Agent
from agents.model import QActor, ParamActor
from utilities.utilities import *


class PDQNBaseAgent(Base_Agent):
    """
    DDPG actor-critic agent for parameterized action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = 'P-DQN Agent'

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.device = torch.device(self.hyperparameters['device'])
        self.action_space = self.environment.action_space
        self.num_actions = self.env_parameters['phase_num']
        # it's decided by env's action_space
        # In FreewheelingIntersection_v0, the action_space is
        # self.action_space = spaces.Tuple((
        #   spaces.Discrete(self.phase_num),
        #   spaces.Tuple(
        #        tuple(spaces.Box(action_low[i], action_high[i], dtype=np.float32) for i in range(self.phase_num))
        #    )
        # ))
        # In this case(FreewheelingIntersection), every continuous action just has one dimension!

        self.action_parameter_size = self.num_actions
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(self.device)
        self.action_min = - self.action_max.detach()  # remove gradient
        self.action_range = (self.action_max - self.action_min).detach()  # 是否要进行归一化

        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high + 1
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)

        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']

        self.initial_memory_threshold = self.hyperparameters['initial_memory_threshold']
        self.batch_size = self.hyperparameters['batch_size']

        self.gamma = self.hyperparameters['gamma']

        self.learning_rate_actor = self.hyperparameters['learning_rate_actor']
        self.learning_rate_actor_param = self.hyperparameters['learning_rate_actor_param']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.tau_actor_param = self.hyperparameters['tau_actor_param']
        self.hidden_layer_actor = self.hyperparameters['hidden_layer_actor']
        self.hidden_layer_actor_param = self.hyperparameters['hidden_layer_actor_param']
        self.clip_grad = self.hyperparameters['clip_grad']

        self.actions_count = 0

        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size,
                                                  mu=0., theta=0.15, sigma=0.0001)

        # ----  Instantiation  ----
        self.state_size = self.env_parameters['phase_num'] * self.env_parameters['pad_length'] * 2
        self.actor = QActor(self.state_size, self.num_actions, self.action_parameter_size
                            , self.hidden_layer_actor).to(self.device)
        self.actor_target = QActor(self.state_size, self.num_actions, self.action_parameter_size,
                                   self.hidden_layer_actor).to(self.device)
        hard_update(source=self.actor, target=self.actor_target)
        self.actor_target.eval()

        self.actor_param = ParamActor(self.state_size, self.num_actions,
                                      self.action_parameter_size, self.hidden_layer_actor_param).to(device)
        self.actor_param_target = ParamActor(self.state_size, self.num_actions,
                                             self.action_parameter_size, self.hidden_layer_actor_param).to(device)
        hard_update(source=self.actor_param, target=self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = self.hyperparameters['loss_func']

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)  # TODO more details
        self.actor_param_optimizer = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)

    def __str__(self):
        desc = super().__str__() + '\n'
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "Epsilon_final: {}\n".format(self.epsilon_final) + \
                "Epsilon_decay: {}\n".format(self.epsilon_decay) + \
                "Loss_func: {}\n".format(self.loss_func) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def ornstein_uhlenbeck_noise(self, all_action_parameters):
        """
        Continuous action exploration using an Ornstein-Uhlenbeck process.

        :param all_action_parameters:
        :return:
        """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def pick_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
                           math.exp(-1. * self.actions_count / self.epsilon_decay)
            self.actions_count += 1
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                all_action_parameters = self.actor_param.forward(state)

                # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter
                # exploration
                if random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)
                    all_action_parameters = torch.from_numpy(np.random.randint(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
                else:
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                self.ornstein_uhlenbeck_noise(all_action_parameters)
                action_parameters = all_action_parameters[action]
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                all_action_parameters = self.actor_param.forward(state)
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().data.numpy()
                action = np.argmax(Q_a)
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action]

        return action, action_parameters, all_action_parameters

    def randomly_pick(self):
        """
        if total_steps < randomly_pick_steps, we will execute random selection.
        The random action includes random phase and all phases' duration.

        :return:
        """
        random_action = [np.random.randint(self.num_actions)]
        random_action_ = [np.random.randint(self.action_space.spaces[i].high) for i in range(1, self.num_actions + 1)]
        return np.concatenate((random_action, random_action_), axis=1)

    def optimize_td_loss(self, memory):
        """
        Mainly based on https://github.com/X-I-N/my_PDQN/blob/main/agent.py

        :return:
        """
        if len(memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()  # int64
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        # 这边多嘴一句，squeeze()是一个降维的作用，最后为[batch_size]
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device).squeeze()

        # ----------------------------- optimize Q-network ------------------------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # 首先torch.max会返回一个nametuple(val, inx)因此[0],又因为keepdim=True,所以最终的size会和input一样，除了
            # 那个max的维度大小变为1，因此需要做一个squeeze()的操作

            # compute the TD error
            target = rewards + (1 - dones) * self.gamma * Qprime

        # compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        # 这边很重要
        # 假设q_values = tensor([[1.1, 1.3],
        #                        [5.1, 9.2]])
        # action.view(-1, 1) = tensor([[0],
        #                              [1]])
        # 那最终的y_predicted = tensor([[1.1],
        #                              [9.2]])
        # 暂时要求记住dim与index是一致的

        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimizer.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        # ------------------------------ optimize ParamActor --------------------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q = self.actor(states, action_params)
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, 1))

        # self.actor.zero_grad()
        # Q_loss.backward()
        # delta_a = deepcopy(action_params.grad.data)

        # action_params = self.actor_param(states)
        # delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        # out = -torch.mul(delta_a, action_params)  # Multiplies input by other
        # self.actor_param.zero_grad()
        # out.backward(torch.ones(out.shape)).to(self.device)

        self.actor_param_optimizer.zero_grad()
        Q_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimizer.step()

        soft_update(source=self.actor, target=self.actor_target, tau=self.tau_actor)
        soft_update(source=self.actor_param, target=self.actor_param_target, tau=self.tau_actor_param)

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.actor_param.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(actor_path, actor_param_path))
        self.actor_param.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass
