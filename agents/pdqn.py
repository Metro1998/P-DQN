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

    def forward(self, state, action_parameters):
        negative_slope = 0.01  # TODO

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == 'relu':
                x = F.relu(self.layers[i](x))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError('Unknown activation function' + str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type='kaiming', activation='relu', init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function  # TODO
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        input_size = self.state_size  # TODO
        last_hidden_layers_size = input_size
        if hidden_layers is not None:
            num_hidden_layers = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, num_hidden_layers):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_layers_size = hidden_layers[num_hidden_layers - 1]
        self.action_parameters_output_layer = nn.Linear(last_hidden_layers_size, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)  # TODO

        # initialize layer weights
        for i in range(len(self.layers)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
            elif init_type == 'normal':
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type " + str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix pass_through layer to avoid instability, rest of network can compensate  # TODO
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(num_hidden_layers):
            if self.activation == 'relu':
                F.relu(self.layers[i](x))
            elif self.activation == 'leaky_relu':
                F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_parameter = self.action_parameters_output_layer(x)
        action_parameter += self.action_parameters_passthrough_layer(state)

        if self.squashing_function:  # TODO
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
            # action_params = action_params / torch.norm(action_params) ## REMOVE --- normalisation layer?? for
            # pointmass
        return action_params


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterized action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = 'P-DQN Agent'

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # TODO
                 tau_actor_param=0.01,  # TODO
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,  # TODO
                 use_ornstein_noise=False,
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=False,  # TODO
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device='cuda'if torch.cuda.is_available() else 'cpu',
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n  # TODO
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0]
                                                for i in range(1, self.num_actions + 1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())



