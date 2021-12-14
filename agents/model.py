# @author Metro
# @time 2021/11/3

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layer=(256, 128, 64)):
        """

        :param state_size:
        :param action_size:
        :param action_parameter_size:
        :param hidden_layer:
        """
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        input_size = self.state_size + self.action_parameter_size
        last_hidden_layers_size = input_size
        if hidden_layer is not None:
            num_hidden_layers = len(hidden_layer)
            self.layers.append(nn.Linear(input_size, hidden_layer[0]))
            for i in range(1, num_hidden_layers):
                self.layers.append(nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
            last_hidden_layers_size = hidden_layer[num_hidden_layers - 1]
        self.layers.append(nn.Linear(last_hidden_layers_size, self.action_size))

    def forward(self, state, action_parameters):

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            x = F.relu(self.layers[i](x))
        Q_value = self.layers[-1](x)
        return Q_value


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers):
        """

        :param state_size:
        :param action_size:
        :param action_parameter_size:
        :param hidden_layers:
        """
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        input_size = self.state_size
        last_hidden_layers_size = input_size
        if hidden_layers is not None:
            num_hidden_layers = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, num_hidden_layers):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_layers_size = hidden_layers[num_hidden_layers - 1]
        self.action_parameters_output_layer = nn.Linear(last_hidden_layers_size, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)  # TODO

        # fix pass_through layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        num_hidden_layers = len(self.layers)
        for i in range(num_hidden_layers):
            x = F.relu(self.layers[i](x))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        return action_params
