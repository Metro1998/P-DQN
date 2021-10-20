"""
The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os

from absl import logging
import numpy as np
import tensorflow as tf

import gin.tf

# Defines a type describing of the tuple returned by the replay memory
# Each element of the tuple is a tensor of shape[batch, ...] where ... is defined the 'shape' field of ReplayElement.
# The tensor type is given by the 'type' field.
# The 'name' field is for convenience and ease of debugging.  # TODO

ReplayElement = (collections.namedtuple('shape_type', ['name', 'shape', 'type']))

# A prefix that can not collide with variable names for checkpoint files.  # TODO
STORE_FILENAME_PREFIX = '$store$_'

# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4

def invalid_range()  # TODO

class OutOfGraphReplayBuffer(object):
    """
    A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    Attributes:
        add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
        invalid_range: np.array, an array with the indices of cursor-related invalid transitions
    """

    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        """

        :param observation_shape: tuple of ints.
        :param stack_size: int, number of frames to use in state stack.  # TODO
        :param replay_capacity: int, number of transitions to keep in memory.
        :param batch_size: int.
        :param update_horizon: int, length of update ('n' in n-step update).
        :param gamma: int, the discount factor.
        :param max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.  # TODO
        :param extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
        :param observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
        :param terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
        :param action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
        :param action_dtype: np.dtype, type of elements in the action.
        :param reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
        :param reward_dtype: np.dtype, type of elements in the reward.
        :raise ValueError: If replay_capacity is too small to hold at least one transition.
        """
        assert isinstance(observation_shape, tuple)
        if replay_capacity < update_horizon + stack_size:
            raise ValueError('There is not enough capacity to cover update_horizon and stack_size.')

        logging.info(
            'Creating a %s replay memory with the following parameters:', self.__class__.__name__
        )
        logging.info('\t observation_shape: %s', str(observation_shape))
        logging.info('\t observation_dtype: %s', str(observation_dtype))
        logging.info('\t terminal_dtype: %s', str(terminal_dtype))
        logging.info('\t stack_size: %d', stack_size)
        logging.info('\t replay_capacity: %d', replay_capacity)
        logging.info('\t batch_size: %d', batch_size)
        logging.info('\t update_horizon: %d', update_horizon)
        logging.info('\t gamma: %f', gamma)

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape + (self._stack_size,)  # TODO
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_stroge_types = extra_storage_types
        else:
            self._extra_stroge_types = []
        self._create_stroge()
        self.add_count = np.array(0)
        """
        self.invalid_range = np.zeros((self._stack_size))
    # When the horizon is > 1, we compute the sum of discounted rewards as a dot
    # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon)],
        dtype=np.float32)
    self._next_experience_is_episode_start = True
    self._episode_end_indices = set()
        """


    def _create_storage(self):
        """
        Creates the numpy arrays used to store transitions.

        :return:
        """
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            # i.e. array_shape = [10000] + list(tuple([2, 1]))
            # array_shape = [10000, 2, 1]
            self._store[storage_element.name] = np.empty(array_shape, dtype=storage_element.type)

    def get_storage_signature(self):
       """
       Returns a default list of elements to be stored in this replay memory.

       :return: list of ReplayElements defining the type of contents stored.
       """
       storage_elements = [
           ReplayElement('observation', self._observation_shape, self._observation_dtype)
           ReplayElement('action', self._action_shape, self._action_dtype),
           ReplayElement('reward', self._reward_shape, self._reward_dtype),
           # TODO
           ReplayElement('terminal', (), self._terminal_dtype)
       ]
       for extra_replay_element in self._extra_stroge_types:
           storage_elements.append(extra_replay_element)
       return storage_elements








