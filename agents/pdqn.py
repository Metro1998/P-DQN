# @author Metro
# @time 2021/10/29

"""
  Ref https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
      https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
"""

import math
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.base_agent import Base_Agent
from agents.net import DuelingDQN, GaussianPolicy
from utilities.utilities import *


class P_DQN(Base_Agent):
    """
    A soft actor-critic agent for hybrid action spaces

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
        self.lr_alpha = self.hyperparameters['lr_alpha']
        self.tau_critic = self.hyperparameters['tau_critic']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.critic_hidden_layers = self.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = self.hyperparameters['actor_hidden_layers']

        self.counts = 0
        self.alpha = 0.2

        # critic
        self.critic = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                 ).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.critic_target = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.critic, target=self.critic_target)

        # actors
        state_dim_actor = int(self.state_dim / 4)
        action_dim_actor = int(self.action_dim / 8)
        self.actor_st = GaussianPolicy(
            state_dim_actor, action_dim_actor, self.actor_hidden_layers, ).to(self.device)
        self.actor_le = GaussianPolicy(
            state_dim_actor, action_dim_actor, self.actor_hidden_layers, ).to(self.device)
        self.actor_sl = GaussianPolicy(
            state_dim_actor, action_dim_actor, self.actor_hidden_layers, ).to(self.device)
        self.actor_st_optim = optim.Adam(self.actor_st.parameters(), lr=self.lr_actor)
        self.actor_le_optim = optim.Adam(self.actor_le.parameters(), lr=self.lr_actor)
        self.actor_sl_optim = optim.Adam(self.actor_sl.parameters(), lr=self.lr_actor)

        self.target_entropy = -torch.Tensor([self.action_dim]).to(self.device).item()
        self.log_alpha = torch.tensor(-np.log(self.action_dim), dtype=torch.float32, requires_grad=True,
                                      device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_critic)  # todo

    def select_action(self, state, train=True):
        """

        :param state:
        :param train:
        :return:
        """
        self.epsilon_step()
        if train:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # (state_dim) -> (1, state_dim)
                action_params, _ = self.actors_sample(state)

                if random.random() < self.epsilon:
                    Q_a, _ = self.critic(state, action_params)
                    print(Q_a)
                    Q_a = Q_a.detach().cpu().numpy()
                    action = int(np.argmin(Q_a))

                else:
                    Q_a, _ = self.critic(state, action_params)
                    print(Q_a)
                    Q_a = Q_a.detach().cpu().numpy()
                    action = int(np.argmax(Q_a))
                action_params = action_params.squeeze().detach().cpu().numpy()
        # else:
        #    with torch.no_grad():
        #       _, _, action_params = self.actor.sample(state) # TODO
        #       Q_a = self.critic.forward(state, action_params)
        #      Q_a = Q_a.detach().cpu().numpy()
        #       action = int(np.argmax(Q_a))
        #       action_params = action_params.detach().cpu().numpy()

        return action, action_params

    def update(self, memory, actor_name, batch_size):
        if len(memory) < batch_size:
            batch_size = len(memory)
        state_batch, action_batch, action_params_batch, reward_batch, next_state_batch, done_batch = memory.sample(
            batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.IntTensor(action_batch).to(self.device).long().unsqueeze(1)
        action_params_batch = torch.FloatTensor(action_params_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # ------------------------------------ update critic -----------------------------------------------
        with torch.no_grad():
            next_state_action_params, next_state_log_pi = self.actors_sample(next_state_batch)
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action_params)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            min_q_next_target = torch.max(min_q_next_target, 1, keepdim=True)[0].squeeze()
            q_next = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        q1, q2 = self.critic(state_batch, action_params_batch)
        q1 = q1.gather(1, action_batch.view(-1, 1)).squeeze()
        q2 = q2.gather(1, action_batch.view(-1, 1)).squeeze()
        q_loss = F.mse_loss(q1, q_next) + F.mse_loss(q2, q_next)

        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), [0., 20.])
        self.critic_optim.step()
        soft_update(self.critic_target, self.critic, self.tau_critic)

        # ------------------------------------ update actor -----------------------------------------------
        # pi, log_pi, _ = self.actor.sample(state_batch)
        """
        if actor_name == 'actor_st':
            pi, log_pi, _ = self.actor_st.sample(state_batch)
        elif actor_name == 'actor_le':
            pi, log_pi, _ = self.actor_le.sample(state_batch)
        elif actor_name == 'actor_sl':
            pi, log_pi, _ = self.actor_sl.sample(state_batch)
        else:
            return 'Invalid actor_name'
        """
        with torch.no_grad():
            pi, log_pi = self.actors_sample(state_batch)
        pi.requires_grad = True
        # pi = torch.gather(pi, 1, action_batch)  # action_batch(batch_size, 1)
        # log_pi = torch.gather(log_pi, 1, action_batch)

        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi.gather(1, action_batch), q2_pi.gather(1, action_batch))
        # min_q_pi = torch.min(q1_pi.mean(), q2_pi.mean())

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # self.actor_optim.step()
        if actor_name == 'actor_st':
            self.actor_st_optim.zero_grad()
            actor_loss.backward()
            self.actor_st_optim.step()
        elif actor_name == 'actor_le':
            self.actor_le_optim.zero_grad()
            actor_loss.backward()
            self.actor_le_optim.step()
        elif actor_name == 'actor_sl':
            self.actor_sl_optim.zero_grad()
            actor_loss.backward()
            self.actor_sl_optim.step()
        else:
            return 'Invalid actor_name'

        # ------------------------------------ update alpha -----------------------------------------------
        alpha_loss = (self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.detach().exp()

    def actors_sample(self, state):
        """
        Samples through three_body model, actually there is 8 actors

        :param state:FloatTensor(batch_size, action_params_dim)
        :return: pi and log_pi w.r.t 8 stages
        """
        st_NS_a, st_NS_p, _ = self.actor_st.sample(torch.cat((state[:, 0].unsqueeze(1), state[:, 4].unsqueeze(1)), 1))  # batch_size * 2
        st_EW_a, st_EW_p, _ = self.actor_st.sample(torch.cat((state[:, 2].unsqueeze(1), state[:, 6].unsqueeze(1)), 1))
        le_NS_a, le_NS_p, _ = self.actor_le.sample(torch.cat((state[:, 1].unsqueeze(1), state[:, 5].unsqueeze(1)), 1))
        le_EW_a, le_EW_p, _ = self.actor_le.sample(torch.cat((state[:, 3].unsqueeze(1), state[:, 7].unsqueeze(1)), 1))
        sl_N_a, sl_N_p, _ = self.actor_sl.sample(torch.cat((state[:, 0].unsqueeze(1), state[:, 1].unsqueeze(1)), 1))
        sl_E_a, sl_E_p, _ = self.actor_sl.sample(torch.cat((state[:, 2].unsqueeze(1), state[:, 3].unsqueeze(1)), 1))
        sl_S_a, sl_S_p, _ = self.actor_sl.sample(torch.cat((state[:, 4].unsqueeze(1), state[:, 5].unsqueeze(1)), 1))
        sl_W_a, sl_W_p, _ = self.actor_sl.sample(torch.cat((state[:, 6].unsqueeze(1), state[:, 7].unsqueeze(1)), 1))
        action_params = torch.cat((st_NS_a, st_EW_a, le_NS_a, le_EW_a,
                                   sl_N_a, sl_E_a, sl_S_a, sl_W_a), 1)
        log_prob = torch.cat((st_NS_p, st_EW_p, le_NS_p, le_EW_p,
                              sl_N_p, sl_E_p, sl_S_p, sl_W_p), 1)

        return action_params, log_prob

    def save_models(self, critic_path, actor_path):
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor_st.state_dict(), actor_path)
        torch.save(self.actor_le.state_dict(), actor_path)
        torch.save(self.actor_sl.state_dict(), actor_path)
        print('Models saved successfully')

    def load_models(self, critic_path, actor_path):
        # also try load on CPU if no GPU available?
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor_st.load_state_dict(torch.load(actor_path))
        self.actor_st.load_state_dict(torch.load(actor_path))
        self.actor_st.load_state_dict(torch.load(actor_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def epsilon_step(self):
        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(
            -1. * self.counts / self.epsilon_decay)
        self.counts += 1