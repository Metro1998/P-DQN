from utilities.data_structure.node import Node
import numpy as np

deque = np.array([Node(0, tuple([None for _ in range(2)]))
                  for _ in range(10)])
print(deque)