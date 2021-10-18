import numpy as np
from utilities.data_structure.node import Node


class Max_Heap(object):
    """ Generic max heap object. """
    def __init__(self, max_size, dimension_of_value_attributue, default, default_key_to_use):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attributue
        self.default_key_to_use = default_key_to_use
        

