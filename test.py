import torch
import torch.nn as nn

m = nn.Embedding(200, 5)

input_ = torch.tensor([[1, 4, 9], [2, 6, 3]])

out_put_ = m(input_)

print(out_put_)


