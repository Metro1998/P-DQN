# @author Metro
# @time 2021/10/29

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
"""

import math

import numpy as np
import torch.optim as optim
import random
from agents.base_agent import Base_Agent
from agents.net_0 import DuelingDQN, GaussianPolicy
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

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']

        self.initial_memory_threshold = self.hyperparameters['initial_memory_threshold']
        self.batch_size = self.hyperparameters['batch_size']

        self.gamma = self.hyperparameters['gamma']

        self.learning_rate_QNet = self.hyperparameters['learning_rate_QNet']
        self.learning_rate_ParamNet = self.hyperparameters['learning_rate_ParamNet']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.tau_actor_param = self.hyperparameters['tau_actor_param']
        self.hidden_layers = self.hyperparameters['adv_hidden_layers']
        self.param_hidden_layers = self.hyperparameters['param_hidden_layers']
        self.clip_grad = self.hyperparameters['clip_grad']

        self.counts = 0

        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size,
                                                  mu=0., theta=0.15, sigma=0.0001)

        # ----  Initialization  ----
        self.state_dim = self.env_parameters['phase_num'] * self.env_parameters['cells'] * 2
        self.param_state_dim = self.env_parameters['phase_num']
        self.Critic = DuelingDQN(self.state_dim, self.num_actions, self.param_state_dim, self.hidden_layers,
                                 ).to(self.device)
        self.Critic_target = DuelingDQN(self.state_dim, self.num_actions, self.param_state_dim, self.hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.Critic, target=self.Critic_target)
        self.Critic_target.eval()

        self.Actor = GaussianPolicy(self.state_dim, self.num_actions,
                                    self.param_hidden_layers).to(self.device)
        self.Actor_target = GaussianPolicy(self.state_dim, self.num_actions,
                                           self.param_hidden_layers).to(self.device)
        hard_update(source=self.Actor, target=self.Actor_target)
        self.Actor_target.eval()

        self.loss_func = self.hyperparameters['loss_func']
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=self.learning_rate_QNet)  # TODO more details
        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=self.learning_rate_ParamNet)

    def __str__(self):
        desc = super().__str__() + '\n'
        desc += "Actor Network {}\n".format(self.Critic) + \
                "Param Network {}\n".format(self.Actor) + \
                "Actor Alpha: {}\n".format(self.learning_rate_QNet) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_ParamNet) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "Epsilon_final: {}\n".format(self.epsilon_final) + \
                "Epsilon_decay: {}\n".format(self.epsilon_decay) + \
                "Loss_func: {}\n".format(self.loss_func) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def pick_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * \
                           math.exp(1. * self.counts / self.epsilon_decay)  # TODO
            self.counts += 1
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                all_action_params = self.Actor.forward(state)
                print('all_action_params:', all_action_params)

                # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter
                # exploration
                if random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)
                    all_action_params = torch.FloatTensor(np.random.randint(low=5, high=15 + 1,
                                                                            size=self.num_actions))
                else:
                    Q_a = self.Critic.forward(state.unsqueeze(0), all_action_params.unsqueeze(0))
                    print('Q_a', Q_a)
                    Q_a = Q_a.detach().data.numpy()
                    action = np.argmax(Q_a)

                all_action_params = all_action_params.cpu().data.numpy()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                all_action_params = self.Actor.forward(state)
                Q_a = self.Critic.forward(state.unsqueeze(0), all_action_params.unsqueeze(0))
                Q_a = Q_a.detach().data.numpy()
                action = np.argmax(Q_a)
                all_action_params = all_action_params.cpu().data.numpy()

        return action, all_action_params

    def optimize_td_loss(self, memory):
        """
        Mainly based on https://github.com/X-I-N/my_PDQN/blob/main/agent.py

        :return:
        """
        if len(memory) < self.batch_size:
            return
        states, actions, all_action_params, rewards, next_states, dones = memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).squeeze()
        all_action_params = torch.FloatTensor(all_action_params).to(self.device).squeeze()
        rewards = torch.FloatTensor(rewards).to(self.device).squeeze()
        # 这边多嘴一句，squeeze()是一个降维的作用，最后为[batch_size]
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).squeeze()

        # ----------------------------- optimize Q-network ------------------------------------
        with torch.no_grad():
            pred_next_action_parameters = self.Actor_target.forward(next_states)
            pred_Q_a = self.Critic_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # 首先torch.max会返回一个nametuple(val, inx)因此[0],又因为keepdim=True,所以最终的size会和input一样，除了
            # 那个max的维度大小变为1，因此需要做一个squeeze()的操作

            # compute the TD error
            target = rewards + (1 - dones) * self.gamma * Qprime

        # compute current Q-values using policy network
        q_values = self.Critic(states, all_action_params)
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

        self.Critic_optimizer.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), self.clip_grad)
        self.Critic_optimizer.step()

        # ------------------------------ optimize ParamActor --------------------------------
        with torch.no_grad():
            action_params = self.Actor(states)
        action_params.requires_grad = True
        Q_val = self.Critic(states, action_params)
        Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
        Q_loss = - torch.mean(Q_indexed)

        # self.actor.zero_grad()
        # Q_loss.backward()
        # delta_a = deepcopy(action_params.grad.data)

        # action_params = self.actor_param(states)
        # delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        # out = -torch.mul(delta_a, action_params)  # Multiplies input by other
        # self.actor_param.zero_grad()
        # out.backward(torch.ones(out.shape)).to(self.device)

        self.Actor_optimizer.zero_grad()
        Q_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_grad)
        self.Actor_optimizer.step()

        soft_update(source=self.Critic, target=self.Critic_target, tau=self.tau_actor)
        soft_update(source=self.Actor, target=self.Actor_target, tau=self.tau_actor_param)

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.Critic.state_dict(), actor_path)
        torch.save(self.Actor.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.Critic.load_state_dict(torch.load(actor_path, actor_param_path))
        self.Actor.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass
