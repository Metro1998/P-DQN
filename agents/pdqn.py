import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable

from agents.agent import Agent
from agents.memory.memory import Memory
from agents.model import Critic, Actor
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise


class P_DQN(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self, config=None):
        super(P_DQN, self).__init__(env, config)

        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, self.num_actions + 1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(self.device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        print([self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)])
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)

        self.epsilon = config.hyperparameters['epsilon_initial']
        self.epsilon_initial = config.hyperparameters['epsilon_initial']
        self.epsilon_final = config.hyperparameters['epsilon_final']
        self.epsilon_decay = config.hyperparameters['epsilon_decay']
        self.indexed = config.others['indexed']
        self.weighted = config.others['weighted']
        self.average = config.others['average']
        self.random_weighted = config.others['random_weighted']
        assert (self.weighted ^ self.average ^ self.random_weighted) or not (self.weighted or self.average or self.random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = config.hyperparameters['batch_size']
        self.gamma = config.hyperparameters['gamma']
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.initial_memory_threshold = config.hyperparameters['initial_memory_threshold']
        self.lr_critic = config.hyperparameters['lr_critic']
        self.lr_actor = config.hyperparameters['lr_actor']
        self.inverting_gradients = config.others['inverting_gradients']
        self.tau_critic = config.hyperparameters['tau_critic']
        self.tau_actor = config.hyperparameters['tau_actor']
        self.clip_grad = config.hyperparameters['clip_grad']
        self.zero_index_gradients = config.others['zero_index_gradients']
        self.critic_hidden_layers = config.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = config.hyperparameters['actor_hidden_layers']
        self.init_std = config.hyperparameters['init_std']

        self._step = 0
        self._episode = 0
        self.updates = 0

        self.use_ornstein_noise = config.use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.local_rnd, mu=0.,
                                                  theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)

        self.replay_memory = Memory(self.replay_memory_size, self.observation_space.shape,
                                    (1 + self.action_parameter_size, ), next_actions=False)
        self.critic = Critic(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                             self.critic_hidden_layers, self.init_std).to(self.device)
        self.critic_target = Critic(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                    self.critic_hidden_layers, self.init_std).to(self.device)
        hard_update_target_network(self.critic, self.critic_target)
        self.critic_target.eval()

        self.actor = Actor(self.observation_space.shape[0], self.action_parameter_size, self.actor_hidden_layers,
                           self.init_std).to(self.device)
        self.actor_target = Actor(self.observation_space.shape[0], self.action_parameter_size, self.actor_hidden_layers,
                                  self.init_std).to(self.device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.loss_func = config.hyperparameters['loss_func']  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)  # , betas=(0.95, 0.999))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)  # , betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.critic) + \
                "Param Network {}\n".format(self.actor) + \
                "Actor Alpha: {}\n".format(self.lr_critic) + \
                "Actor Param Alpha: {}\n".format(self.lr_actor) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_critic) + \
                "Tau (actor-params): {}\n".format(self.tau_actor) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor.action_parameters_passthrough_layer
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor, self.actor_target)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(
            -1. * self._episode / self.epsilon_decay)
        self._episode += 1

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor.forward(state)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.local_rnd.uniform()
            if rnd < self.epsilon:
                action = self.local_rnd.choice(self.num_actions)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
            else:
                # select maximum action
                Q_a = self.critic.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += \
                    self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a + 1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self, memory):
        if len(memory) < self.batch_size:
            batch_size = len(memory)
        else:
            batch_size = self.batch_size

        state_batch, action_batch, action_params_batch, reward_batch, next_state_batch, done_batch = memory.sample(
            batch_size, random_machine=self.local_rnd)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.IntTensor(action_batch).to(self.device).long().unsqueeze(1)
        action_params_batch = torch.FloatTensor(action_params_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_target.forward(next_state_batch)
            pred_Q_a = self.critic_target(next_state_batch, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = reward_batch + (1 - done_batch) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.critic(state_batch, action_params_batch)
        y_predicted = q_values.gather(1, action_batch.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.critic_optim.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optim.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor(state_batch)
        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
               not (self.weighted or self.average or self.random_weighted)
        Q = self.critic(state_batch, action_params)
        Q_val = Q
        if self.weighted:
            # approximate categorical probability density (i.e. counting)
            counts = Counter(action_batch.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / action_batch.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, action_batch.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.critic.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor(Variable(state_batch))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=action_batch, inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)

        self.actor_optim.step()

        soft_update_target_network(self.critic, self.critic_target, self.tau_critic)
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.critic.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.critic.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')
