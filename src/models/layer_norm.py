import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x    = (x - mean) / (var + self.epsilon).sqrt()
        return self.gamma * x + self.beta