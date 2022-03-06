# @author Metro
# @time 2021/11/3

"""
  Ref: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/net.py
       https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


epsilon = 1e-6


class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64),
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.adv_layers_1 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_1 = nn.Linear(hidden_layers[-1], 1)

        self.adv_layers_2 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_2 = nn.Linear(hidden_layers[-1], 1)

        self.apply(init_)

    def forward(self, state, action_params):
        temp = torch.cat((state, action_params), dim=1)

        x1 = temp
        for i in range(len(self.layers)):
            x1 = F.relu(self.layers[i](x1))
        adv1 = self.adv_layers_1(x1)
        val1 = self.val_layers_1(x1)
        q_duel1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        x2 = temp
        for i in range(len(self.layers)):
            x2 = F.relu(self.layers[i](x2))
        adv2 = self.adv_layers_1(x2)
        val2 = self.val_layers_1(x2)
        q_duel2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)

        return q_duel1, q_duel2


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64), action_space=None,
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.mean_layers = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std_layers = nn.Linear(hidden_layers[-1], action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(init_)

    def forward(self, state):
        x = state

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
        mean = self.mean_layers(x)
        log_std = self.log_std_layers(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # noise = torch.randn_like(mean, requires_grad=True)
        # action = (mean + std * noise).tanh()

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # log_prob = log_std + np.log(np.sqrt(2 * np.pi)) + noise.pow(2).__mul__(0.5)
        # log_prob += (-action.pow(2) + 1.00000001).log()
        # log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob, mean
