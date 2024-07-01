import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum()

class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        max_x = torch.max(x)
        exp_x = torch.exp(x - max_x)
        return exp_x / exp_x.sum()

# Test
data = torch.Tensor([1, 2, 3])
softmax = Softmax()
output = softmax(data)
print(output)

data = torch.Tensor([1, 2, 3])
softmax_stable = SoftmaxStable()
output = softmax_stable(data)
print(output)
