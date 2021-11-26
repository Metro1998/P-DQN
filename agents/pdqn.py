# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""
import math

import torch
import torch.optim as optim
import numpy as np
import random

from torch.autograd import Variable
from copy import deepcopy
from agents.base_agent import Base_Agent
from agents.model import QActor, ParamActor
from utilities.utilities import *


class PDQNBaseAgent(Base_Agent):
    """
    DDPG actor-critic agent for parameterized action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = 'P-DQN Agent'

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.device = torch.device(self.hyperparameters['device'])
        self.action_space = self.environment.action_space

        self.num_actions = self.env_parameters['phase_num']
        # it's decided by env's action_space
        # In FreewheelingIntersection_v0, the action_space is
        # self.action_space = spaces.Tuple((
        #   spaces.Discrete(self.phase_num),
        #   spaces.Tuple(
        #        tuple(spaces.Box(action_low[i], action_high[i], dtype=np.float32) for i in range(self.phase_num))
        #    )
        # ))

        # In this case(FreewheelingIntersection), every continuous action just has one dimension!
        self.action_parameter_size = self.num_actions
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = - self.action_max.detach()  # remove gradient
        self.action_range = (self.action_max - self.action_min).detach()  # 是否要进行归一化
        print([self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)])  # TODO
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low
                                                          for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)

        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)
        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']

        self.initial_memory_threshold = self.hyperparameters['initial_memory_threshold']
        self.batch_size = self.hyperparameters['batch_size']

        self.gamma = self.hyperparameters['gamma']

        self.learning_rate_actor = self.hyperparameters['learning_rate_actor']
        self.learning_rate_actor_param = self.hyperparameters['learning_rate_actor_param']

        self.tau_actor = self.hyperparameters['tau_actor']
        self.tau_actor_param = self.hyperparameters['tau_actor_param']
        self.clip_grad = self.hyperparameters['clip_grad']

        self.hidden_layer_actor = self.hyperparameters['hidden_layer_actor']
        self.hidden_layer_actor_param = self.hyperparameters['hidden_layer_actor_param']

        # Randomization is executed in Base_Agent with self.set_random_seeds(random_seed)

        self.actions_count = 0
        self._steps = 0
        self._updates = 0

        # ----  Instantiation  ----
        self.state_size = self.env_parameters['phase_num'] * self.env_parameters['pad_length'] * 2
        self.actor = QActor(self.state_size, self.num_actions, self.action_parameter_size
                            , self.hidden_layer_actor).to(self.device)
        self.actor_target = QActor(self.state_size, self.num_actions, self.action_parameter_size,
                                   self.hidden_layer_actor).to(self.device)
        hard_update(source=self.actor, target=self.actor_target)
        # self.actor_target = load_state_dict(self.actor_net.state_dict())
        self.actor_target.eval()

        self.actor_param = ParamActor(self.state_size, self.num_actions,
                                      self.action_parameter_size, self.hidden_layer_actor_param).to(device)
        self.actor_param_target = ParamActor(self.state_size, self.num_actions,
                                             self.action_parameter_size, self.hidden_layer_actor_param).to(device)
        hard_update(source=self.actor_param, target=self.actor_param_target)
        # self.actor_param_target.load_state_dict(self.actor_param.state_dict())
        self.actor_param_target.eval()

        self.loss_func = self.hyperparameters['loss_func']  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)  # TODO 这边有细节
        self.actor_param_optimizer = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)

    def __str__(self):
        desc = super().__str__() + '\n'
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_decay: {}\n".format(self.epsilon_decay) + \
                "loss_func: {}\n".format(self.loss_func) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):  # TODO
        """

        :param initial_weights:
        :param initial_bias:
        :return:
        """
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        # directly from state to actor_param
        # [self.state_size, self.action_parameter_size]
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update(source=self.actor_param, target=self.actor_param_target)

    def pick_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
                           math.exp(-1. * self.actions_count / self.epsilon_decay)
            self.actions_count += 1
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                all_action_parameters = self.actor_param.forward(state)

                # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter
                # exploration
                if random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
                else:
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action]
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                all_action_parameters = self.actor_param.forward(state)
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().data.numpy()
                action = np.argmax(Q_a)
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action]

        return action, action_parameters, all_action_parameters

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        if grad_type == 'actions':
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == 'action_parameters':
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError('Unhandled grad_type: {}'.format(str(grad_type)))

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            index = grad > 0
            grad[index] *= (index.float() * (max_p - min_p) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def optimize_td_loss(self, memory):
        """
        Mainly based on https://github.com/X-I-N/my_PDQN/blob/main/agent.py

        :return:
        """
        if len(memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()  # int64
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        # 这边多嘴一句，squeeze()是一个降维的作用，最后为[batch_size]
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device).squeeze()

        # ----------------------------- optimize Q-network ------------------------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # 首先torch.max会返回一个nametuple(val, inx)因此[0],又因为keepdim=True,所以最终的size会和input一样，除了
            # 那个max的维度大小变为1，因此需要做一个sqeeze()的操作

            # compute the TD error
            target = rewards + (1 - dones) * self.gamma * Qprime

        # compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        # 这边很重要
        # 假设q_values = tensor([[1.1, 1.3],
        #                        [5.1, 9.2]])
        # action.view(-1, 1) = tensor([[0],
        #                              [1]])
        # 那最终的y_predicted = tensor([[1.1],
        #                              [9.2]])
        # 暂时要求记住dim与index是一致的

        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimizer.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        # ------------------------------ optimize ParamActor --------------------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q_val = self.actor(states, action_params)
        param_loss = torch.mean(torch.sum(Q_val, 1))
        # 首先是sum部分，这和论文中是一致的，即对于所有K个动作进行加和，mean操作则是对batch_size个数据的处理，loss最后是一个float
        self.actor.zero_grad()
        param_loss.backward()

        # TODO
        delta_a = deepcopy(action_params.grad.data)
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        out = -torch.mul(delta_a, action_params)  # Multiplies input by other
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape)).to(self.device)

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimizer.step()

        soft_update(source=self.actor, target=self.actor_target, tau=self.tau_actor)
        soft_update(source=self.actor_param, target=self.actor_param_target, tau=self.tau_actor_param)

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.actor_param.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(actor_path, actor_param_path))
        self.actor_param.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps):  # TODO
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass
