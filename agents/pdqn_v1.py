# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import math

import numpy as np
import torch.optim as optim
import random
from agents.base_agent import Base_Agent
from agents.net_v1 import DuelingDQN, GaussianPolicy
from utilities.utilities import *


class P_DQN(Base_Agent):
    """
    soft actor-critic agent for parameterized action spaces

    """

    NAME = 'P-DQN Agent'

    def __init__(self, config, env):
        Base_Agent.__init__(self, config)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']

        self.initial_memory_threshold = self.hyperparameters['initial_memory_threshold']
        self.batch_size = self.hyperparameters['batch_size']

        self.gamma = self.hyperparameters['gamma']
        self.alpha = 1.  # TODO

        self.lr_critic = self.hyperparameters['learning_rate_QNet']
        self.lr_actor = self.hyperparameters['learning_rate_ParamNet']
        self.tau_critic = self.hyperparameters['tau_actor']
        self.tau_actor = self.hyperparameters['tau_actor_param']
        self.hidden_layers = self.hyperparameters['adv_hidden_layers']

        self.counts = 0
        self.updates = 2

        # ----  Initialization  ----
        self.critic = DuelingDQN(self.state_dim, self.action_dim, self.hidden_layers,
                                 ).to(self.device)
        self.critic_target = DuelingDQN(self.state_dim, self.action_dim, self.hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.critic, target=self.critic_target)
        self.critic_target.eval()

        self.actor = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_layers, env.action_space[0]
                                    ).to(self.device)
        self.actor_target = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_layers, env.action_space[0]
                                           ).to(self.device)
        hard_update(source=self.actor, target=self.actor_target)
        self.actor_target.eval()

        self.alpha_log = torch.FloatTensor(-np.log(self.action_dim) * np.e).to(self.device)
        self.target_entropy = np.log(self.action_dim)

        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.alpha_optimizer = optim.Adam(self.alpha_log, lr=self.lr_critic)  # TODO

    def pick_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
                           math.exp(1. * self.counts / self.epsilon_decay)  # TODO
            self.counts += 1
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_params = self.actor.forward(state)
                print('action_params:', action_params)

                if random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)

                else:
                    Q_a = self.critic.forward(state.unsqueeze(0), action_params.unsqueeze(0))
                    print('Q_a', Q_a)
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)

                action_params = action_params.detach().data.numpy()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_params = self.actor.forward(state)
                Q_a = self.critic.forward(state.unsqueeze(0), action_params.unsqueeze(0))
                Q_a = Q_a.detach().data.numpy()
                action = np.argmax(Q_a)
                action_params = action_params.detach().data.numpy()

        return action, action_params

    def update_net(self, memory):
        """
        Mainly based on https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentSAC.py

        :return:
        """
        if len(memory) > self.batch_size:
            for i in range(self.updates):
                """objective of critic (loss function of critic)"""
                obj_critic, states = self.get_obj_critic(memory)
                self.optim_update(self.critic_optimizer, obj_critic)
                soft_update(self.critic_target, self.critic, self.tau_critic)

                """objective of alpha (temperature parameter automatic adjustment)"""
                action_params, logprob = self.actor.get_action_logprob(states)
                obj_alpha = (
                        self.alpha_log * (logprob - self.target_entropy).detach()
                ).mean()
                self.optim_update(self.alpha_optimizer, obj_alpha)

                """objective of actor"""
                obj_actor = -(self.critic(states, action_params) + logprob * self.alpha_log.exp()).mean()
                self.optim_update(self.actor_optimizer, obj_actor)
                soft_update(self.actor_target, self.critic, self.tau_actor)

    def get_obj_critic(self, memory):
        """
        Calculate the loss of critic networks with **uniform sampling**.

        :param memory:
        :return:
        """
        states, actions, action_params, rewards, next_states, dones = memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        action_params = torch.FloatTensor(action_params).to(self.device).squeeze()
        rewards = torch.FloatTensor(rewards).to(self.device).squeeze()
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).squeeze()
        with torch.no_grad():
            next_action_params, next_log_prob = self.actor_target.get_action_logprob(next_states)
            next_q = torch.min(self.critic_target.get_q1_q2(next_states, action_params=next_action_params))

            q_label = rewards + (1 - dones) * (next_q + next_log_prob * self.alpha)
        q1, q2 = self.critic.get_q1_q2(states, action_params=action_params)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        return obj_critic, states

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.critic.state_dict(), actor_path)
        torch.save(self.actor.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.critic.load_state_dict(torch.load(actor_path, actor_param_path))
        self.actor.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass
