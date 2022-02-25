# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import math

import torch
import torch.optim as optim
import random
from agents.base_agent import Base_Agent
from agents.net import DuelingDQN, GaussianPolicy
from utilities.utilities import *


class P_DQN(Base_Agent):
    """
    soft actor-critic agent for parameterized action spaces

    """

    NAME = 'P-DQN Agent'

    def __init__(self, config, env):
        Base_Agent.__init__(self, config)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space[0].n

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.batch_size = self.hyperparameters['batch_size']
        self.gamma = self.hyperparameters['gamma']

        self.lr_critic = self.hyperparameters['lr_critic']
        self.lr_actor = self.hyperparameters['lr_actor']
        self.tau_critic = self.hyperparameters['tau_critic']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.critic_hidden_layers = self.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = self.hyperparameters['actor_hidden_layers']

        self.counts = 0

        # ----  Initialization  ----
        self.critic = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                 ).to(self.device)
        self.critic_target = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.critic, target=self.critic_target)
        self.critic_target.eval()

        self.actor = GaussianPolicy(self.state_dim, self.action_dim, self.actor_hidden_layers, env.action_space[1]
                                    ).to(self.device)
        self.actor_target = GaussianPolicy(self.state_dim, self.action_dim, self.actor_hidden_layers,
                                           env.action_space[1]).to(self.device)
        hard_update(source=self.actor, target=self.actor_target)
        self.actor_target.eval()

        self.alpha_log = torch.tensor(
            (-np.log(self.action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.target_entropy = np.log(self.action_dim)

        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.alpha_optimizer = optim.Adam([self.alpha_log], lr=self.lr_critic)  # TODO

    def pick_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
                           math.exp(1. * self.counts / self.epsilon_decay)  # TODO
            self.counts += 1
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_params = self.actor.get_action(state)
                print('action_params:', action_params)

                if random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)

                else:
                    Q_a = self.critic.forward(state.unsqueeze(0), action_params.unsqueeze(0))
                    print('Q_a', Q_a)
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)
                action_params = action_params.cpu().data.numpy()
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
            """objective of critic (loss function of critic)"""
            obj_critic, states = self.get_obj_critic(memory, self.alpha_log.exp())
            self.critic_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            obj_critic.backward()

            soft_update(self.critic_target, self.critic, self.tau_critic)

            """objective of alpha (temperature parameter automatic adjustment)"""
            action_params, logprob = self.actor.get_action_logprob(states)
            obj_alpha = (
                    self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(obj_alpha, 10)
            obj_alpha.backward()

            """objective of actor"""
            obj_actor = -(self.critic(states, action_params) + logprob * self.alpha_log.exp()).mean()
            self.actor_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            obj_actor.backward()

    def get_obj_critic(self, memory, alpha):
        """
        Calculate the loss of critic networks with **uniform sampling**.

        :param alpha:
        :param memory:
        :return:
        """
        states, actions, action_params, rewards, next_states, dones = memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().squeeze().to(self.device)
        action_params = torch.from_numpy(action_params).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().squeeze().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).squeeze().to(self.device)

        with torch.no_grad():
            next_action_params, next_log_prob = self.actor_target.get_action_logprob(next_states)
            next_q = torch.min(*self.critic_target.get_q1_q2(next_states, action_params=next_action_params))
            next_q = torch.max(next_q, 1, keepdim=True)[0]

            q_label = rewards + (1 - dones) * (next_q + next_log_prob * alpha)
        q1, q2 = self.critic.get_q1_q2(states, action_params=action_params)
        q1_predicted = q1.gather(1, actions.view(-1, 1))
        q2_predicted = q2.gather(1, actions.view(-1, 1))
        obj_critic = (self.criterion(q1_predicted, q_label) + self.criterion(q2_predicted, q_label)) / 2.0
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
