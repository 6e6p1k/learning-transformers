import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # general parameters
        B, L, D = x.shape
        device = x.device
        i = torch.arange(1, D / 2 + 1, dtype=torch.int32).to(device)
        theta = 1e4 ** (-2 * (torch.stack((i, i), dim=-1).reshape(D) - 1) / D).to(device)
        pos = torch.arange(1, L + 1).unsqueeze(-1).expand(L, D).to(device)

        # part with cos
        cos_applied = x * torch.cos(pos * theta).unsqueeze(0).expand(B, L, D).to(device)
        sin_applied = torch.ones(B, L, D).to(device)

        # part with sin
        sin_applied[:, :, 1::2] = x[:, :, 0::2] * torch.sin(pos[:, 0::2] * theta[0::2]).unsqueeze(0).expand(B, L, D // 2)
        sin_applied[:, :, 0::2] = -x[:, :, 1::2] * torch.sin(pos[:, 1::2] * theta[0::2]).unsqueeze(0).expand(B, L, D // 2)

        return cos_applied + sin_applied