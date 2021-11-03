# @author Metro
# @time 2021/10/29

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from agents.agent import Agent
from utilities.ou_noise import OrnsteinUhlenbeckActionNoise
from utilities.memory.memory import Memory
from utilities.utilities import *


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layer=(100,), action_input_layer=0,
                 activation='relu', output_layer_init_std=None, **kwargs):
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
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

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
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.actor_param_kwargs = actor_param_kwargs
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n  # TODO
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0]
                                                for i in range(1, self.num_actions + 1)])
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

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()  # different from sum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)  # TODO

        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.seed = seed
        self._seed()

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size,
                                                  random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001)

        print(self.num_actions + self.action_parameter_size)
        self.replay_memory = Memory(replay_memory_size, observation_space.shape,
                                    (1 + self.action_parameter_size,), next_actions=False)  # TODO
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size
                                 , **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                        **actor_kwargs).to(device)
        hard_update(source=self.actor, target=self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                             self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                                    self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update(source=self.actor_param, target=self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)  # TODO 这边有细节
        self.actor_param_optimizer = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)

    def _seed(self):
        """

        :return:
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.np_random = np.random.RandomState(self.seed)

        torch.manual_seed(self.seed)
        if self.device == torch.device('cuda'):
            torch.cuda.manual_seed(self.seed)

    def __str__(self):
        desc = super().__str__() + '\n'
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
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



