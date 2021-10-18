import numpy as np
from utilities.data_structure.node import Node


class Max_Heap(object):
    """" Generic max heap object. """

    def __init__(self, max_size, dimension_of_value_attribute, default_key_to_use):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attribute  # 5 in default
        self.default_key_to_use = default_key_to_use
        self.heap = self.initialize_heap()
        self.heap_index_to_overwrite_next = 1

    def initialize_heap(self):
        """ Initialize a heap of Nodes of length self.max_size * 4 + 1. """
        heap = np.array([Node(self.default_key_to_use, tuple([None for _ in range(self.dimension_of_value_attribute)]))
                         for _ in range(self.max_size * 4 + 1)])  # TODO

        # We don't use the 0th element in a heap so we want it to have infinite value
        # so it is never swapped with a lower node.
        # We set self.dimension_of_attribute to be 5 in default
        heap[0] = Node(float("inf"), (None, None, None, None, None))
        return heap

    def update_element_and_reorganize_heap(self, heap_index_for_change, new_element):
        self.update_heap_element(heap_index_for_change, new_element)
        self.reorganize_heap(heap_index_for_change)

    def update_heap_element(self, heap_index, new_element):
        self.heap[heap_index] = new_element

    def reorganize_heap(self, heap_index_changed):
        """
        This reorganize the heap after a new value is added so as to keep the max value at the top of the heap,
        which is index position 1 in the array self.heap.
        For a node which is pushed into the heap, it will go up or down anyway.
        """

        node_key = self.heap[heap_index_changed].key
        parent_index = int(heap_index_changed / 2)

        if node_key > self.heap[parent_index].key:
            self.swap_heap_elements(heap_index_changed, parent_index)
            self.reorganize_heap(parent_index)
        else:
            biggest_child_index = self.calculate_index_of_biggest_child(heap_index_changed)
            if node_key < self.heap[biggest_child_index].key:
                self.swap_heap_elements(heap_index_changed, biggest_child_index)
                self.reorganize_heap(biggest_child_index)

    def swap_heap_elements(self, index1, index2):
        """ Swaps the position of two heap elements. """
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def calculate_index_of_biggest_child(self, heap_index_changed):  # TODO
        """ Calculate the heap index of the node's child with the biggest td_error value. """
        left_child = self.heap[int(heap_index_changed * 2)]
        right_child = self.heap[int(heap_index_changed * 2) + 1]

        if left_child.key > right_child.key:
            biggest_child_index = heap_index_changed * 2
        else:
            biggest_child_index = heap_index_changed * 2 + 1

        return biggest_child_index
