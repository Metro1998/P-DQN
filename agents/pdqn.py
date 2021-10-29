# @author Metro
# @time 2021/10/29

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from agents.agent import Agent


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layer=(100,), action_input_layer=0,
                 activation='relu', output_layer_init_std=None):
        """

        :param state_size: # TODO
        :param action_size:
        :param action_parameter_size:
        :param hidden_layer:
        :param action_input_layer:
        :param activation:
        """
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

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

        # initialize layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0, std=output_layer_init_std)
        else:
            nn.init.normal_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)


