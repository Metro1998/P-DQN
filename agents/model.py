import torch
import torch.nn as nn
import torch.nn.functional as F


def init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Critic(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,),
                 output_layer_init_std=None):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.state_size + self.action_parameter_size, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        lastHiddenLayerSize = hidden_layers[len(hidden_layers) - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        self.apply(init_)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)

    def forward(self, state, action_parameters):
        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            x = F.relu(self.layers[i](x))
        Q_value = self.layers[-1](x)

        return Q_value


class Actor(nn.Module):

    def __init__(self, state_size, action_parameter_size, hidden_layers,
                 output_layer_init_std=None,):
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.state_size, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        lastHiddenLayerSize = hidden_layers[len(hidden_layers) - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)

        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        self.apply(init_)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            x = F.relu(self.layers[i](x))
        action_params = self.action_parameters_output_layer(x)
        # action_params += self.action_parameters_passthrough_layer(state)  # TODO

        return action_params
