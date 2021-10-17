"""
source:
https://github.com/google
/dopamine/blob/master/dopamine/replay_memory/sum_tree.py

A sum tree data structure.

Used for prioritized experience replay. See prioritized_replay_buffer.py
and Schual et al. (2015).
"""

import math
import random
import numpy as np


class SumTree(object):
    """
    A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]
    """

    def __init__(self, capacity):
        """
        Create the sum tree data structure for the given replay capacity.
        :param capacity:int, the maximum number of elements that can be store in this dara structure
        """
        assert isinstance(capacity, int)
        if capacity <= 0:
            raise ValueError('Sum tree capacity should be positive. Got:{}'.
                             format(capacity))
        self.nodes = []
        tree_depth = int(math.ceil(np.log2(capacity)))
        level_size = 1
        for _ in range(tree_depth + 1):
            nodes_at_this_depth = np.zeros(level_size)
            self.nodes.append(nodes_at_this_depth)

            level_size *= 2

        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """
        Return the sum of all priorities stored in this sum tree.
        :return: float, sum of priorities stored in this sum tree.
        """
        return self.nodes[0][0]

    def sample(self, query_value=None):
        """
        Sample an element from the sum tree.

        Each element has probability p_i/ sum_j p_j of being picked,
        where p_i is the (positive) value associated with node_i.
        :param query_value: float in [0, 1], used as the random value to select a sample.
        If none, will select one randomly in [0, 1).
        :return: int, a random element from the sum tree.
        """
        if self._total_priority() == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')

        if query_value and (query_value < 0. or query_value > 1.0):
            raise ValueError('query_value must be in [0, 1].')

        # Sample a value in range[0, R), where R is the value stored at the root.
        query_value = random.random() if query_value is None else query_value
        query_value *= self._total_priority()

        # Now traverse the sum tree.
        node_index = 0
        for nodes_at_this_depth in self.nodes[1:]:
            # Compute children of previous depth's node.
            left_child = node_index * 2

            left_sum = nodes_at_this_depth[left_child]
            # Each subtree describle a range [0, a), where a is its value.
            if query_value < left_sum:  # Recurse into left subtree.
                node_index = left_child
            else:  # Recurse into right subtree.
                node_index = left_child + 1
                # Adjust query to be relative to right subtree.
                query_value -= left_sum

        return node_index





