# @author Metro
# @time 2021/11/3

"""
  Ref: https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    def __init__(self, state_size, action_size, adv_hidden_layers=(256, 128, 64),
                 val_hidden_layers=(256, 128, 64)):
        """

        :param state_size:
        :param action_size:
        :param adv_hidden_layers:
        :param val_hidden_layers
        """
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # create layers
        self.adv_layers = nn.ModuleList()
        self.val_layers = nn.ModuleList()
        input_size = self.state_size + self.action_parameter_size

        # adv_layers
        self.adv_layers.append(nn.Linear(input_size, adv_hidden_layers[0]))
        for i in range(1, len(adv_hidden_layers)):
            self.adv_layers.append(nn.Linear(adv_hidden_layers[i - 1], adv_hidden_layers[i]))
        self.adv_layers.append(nn.Linear(adv_hidden_layers[-1], self.action_size))

        # val_layers
        self.val_layers.append(nn.Linear(input_size, val_hidden_layers[0]))
        for i in range(1, len(val_hidden_layers)):
            self.val_layers.append(nn.Linear(val_hidden_layers[i - 1], val_hidden_layers[i]))
        self.val_layers.append(nn.Linear(val_hidden_layers[-1], 1))

    def forward(self, state, action_parameters):
        # batch_size = x.size(0)
        x = torch.cat((state, action_parameters), dim=1)

        adv = x
        adv_num_layers = len(self.adv_layers)
        for i in range(0, adv_num_layers - 1):
            adv = F.relu(self.adv_layers[i](adv))
        adv = self.adv_layers[-1](adv)

        val = x
        val_num_layers = len(self.val_layers)
        for i in range(0, val_num_layers - 1):
            val = F.relu(self.val_layers[i](val))
        val = self.val_layers[-1](val)

        Q_value = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return Q_value


class ParamNet(nn.Module):

    def __init__(self, param_state_size, action_size, param_hidden_layers):
        """

        :param param_state_size:
        :param action_size:
        :param param_hidden_layers:
        """
        super(ParamNet, self).__init__()

        self.param_state_size = param_state_size
        self.action_size = action_size

        # create layers
        self.layers = nn.ModuleList()
        input_size = self.param_state_size
        self.layers.append(nn.Linear(input_size, param_hidden_layers[0]))
        for i in range(1, len(param_hidden_layers)):
            self.layers.append(nn.Linear(param_hidden_layers[i - 1], param_hidden_layers[i]))
        self.layers.append(nn.Linear(param_hidden_layers[-1], self.action_size))

    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        action_params = torch.sigmoid(self.layers[-1](x)) * 15 + 10

        return action_params
