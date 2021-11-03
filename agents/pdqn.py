# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from agents.agent import Agent
from agents.model import QActor, ParamActor
# from utilities.ou_noise import OrnsteinUhlenbeckActionNoise
from utilities.memory.memory import Memory
from utilities.utilities import *


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterized action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = 'P-DQN Agent'

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_kwargs={},
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_decay=5000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # soft update
                 tau_actor_param=0.01,
                 replay_memory_size=1e6,
                 learning_rate_actor=1e-4,
                 learning_rate_actor_param=1e-5,
                 use_ornstein_noise=False,
                 loss_func=F.smooth_l1_loss,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.actor_param_kwargs = actor_param_kwargs
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n
        # it's decided by env's action_space
        # from from https://github.com/cycraig/gym-soccer/blob/master/gym_soccer/envs/soccer_score_goal.py
        # self.action_space = spaces.Tuple((spaces.Discrete(3),
        #                                   spaces.Box(low=low0, high=high0, dtype=np.float32),
        #                                   spaces.Box(low=low1, high=high1, dtype=np.float32),
        #                                   spaces.Box(low=low2, high=high2, dtype=np.float32)))

        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0]
                                                for i in range(1, self.num_actions + 1)])
        # every discrete action may have more than one continuous actions
        # self.action_parameter_sizes = [2, 1, 2, 1] in the upper specific instance
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = - self.action_max.detach()  # remove gradient
        self.action_range = (self.action_max - self.action_min).detach()
        print([self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)])  # TODO
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)

        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.actions_count = 0
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param

        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)
        if self.device == torch.device('cuda'):
            torch.cuda.manual_seed(self.seed)
        # self._step = 0
        # self._episode = 0
        # self.updates = 0

        print(self.num_actions + self.action_parameter_size)

        self.actor = QActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_size
                            , **actor_kwargs).to(device)
        self.actor_target = QActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                   **actor_kwargs).to(device)
        hard_update(source=self.actor, target=self.actor_target)
        # self.actor_target = load_state_dict(self.actor_net.state_dict())
        self.actor_target.eval()

        self.actor_param = ParamActor(self.observation_space.shape[0], self.num_actions,
                                      self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = ParamActor(self.observation_space.shape[0], self.num_actions,
                                             self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update(source=self.actor_param, target=self.actor_param_target)
        # self.actor_param_target.load_state_dict(self.actor_param.state_dict())
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)  # TODO 这边有细节
        self.actor_param_optimizer = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)
        self.replay_memory = Memory(replay_memory_size, observation_space.shape,
                                    (1 + self.action_parameter_size,), next_actions=False)  # TODO

    def __str__(self):
        desc = super().__str__() + '\n'
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_decay: {}\n".format(self.epsilon_decay) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        """

        :param initial_weights:
        :param initial_bias:
        :return:
        """
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        # directly from state to actor_param
        # [self.state_size, self.action_parameter_size]
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update(source=self.actor_param, target=self.actor_param_target)

    def choose_action(self, state, train=True):
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
                    action = self.np_random.choice(self.num_actions)
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
                else:
                    # select maximum action
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    # 可能因为forward里面有个cat操作
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)

                # add noise only to parameters of chosen action
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                # offset will help you find the exactly action_parameters
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                all_action_parameters = self.actor_param.forward(state)
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().data.numpy()
                action = np.argmax(Q_a)
                all_action_parameters = all_action_parameters.cpu().data.numpy()  # TODO cpu() detach()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_size[action]]

        return action, action_parameters, all_action_parameters

    def update(self):
        """
        Mainly based on https://github.com/X-I-N/my_PDQN/blob/main/agent.py

        :return:
        """
        if len(self.replay_memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()  # TODO
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()  # TODO
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        # ----------------------------- optimize Q-network ------------------------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()  # TODO

            # compute the TD error
            target = rewards + (1 - dones) * self.gamma * Qprime

        # compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimizer.zero_grad()
        loss_Q.backward()
        for param in self.actor.parameters():
            param.grad.clamp_(-1, 1)
        self.actor_optimizer.step()



