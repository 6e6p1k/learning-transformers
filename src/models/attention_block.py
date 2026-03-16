import torch
from torch import nn
from src.models.RoPE import RoPE

class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0
        self.num_heads = num_heads
        self.heads_dim = embedding_size // num_heads

        self.W_q = nn.Linear(embedding_size, embedding_size)
        self.W_k = nn.Linear(embedding_size, embedding_size)
        self.W_v = nn.Linear(embedding_size, embedding_size)
        self.W_o = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        # (B, num_heads, L, heads_dim)

        # causal mask — upper triangle = -inf so future tokens are invisible
        QK_T = torch.matmul(Q, K.transpose(-1, -2)) / (self.heads_dim ** 0.5)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        QK_T = QK_T.masked_fill(mask, float("-inf"))
        attn = torch.softmax(QK_T, dim=-1)
        # (B, num_heads, L, L)

        out  = torch.matmul(attn, V)
        # (B, num_heads, L, heads_dim)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
    

class AttentionBlockWithRoPE(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0
        self.num_heads = num_heads
        self.heads_dim = embedding_size // num_heads

        self.W_q = nn.Linear(embedding_size, embedding_size)
        self.W_k = nn.Linear(embedding_size, embedding_size)
        self.W_v = nn.Linear(embedding_size, embedding_size)
        self.W_o = nn.Linear(embedding_size, embedding_size)
        self.rope = RoPE()

    def forward(self, x):
        B, L, D = x.shape

        Q = self.rope(self.W_q(x)).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        K = self.rope(self.W_k(x)).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.heads_dim).transpose(1, 2)
        # (B, num_heads, L, heads_dim)


        # causal mask — upper triangle = -inf so future tokens are invisible
        QK_T = torch.matmul(Q, K.transpose(-1, -2)) / (self.heads_dim ** 0.5)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        QK_T = QK_T.masked_fill(mask, float("-inf"))
        attn = torch.softmax(QK_T, dim=-1)
        # (B, num_heads, L, L)

        out  = torch.matmul(attn, V)
        # (B, num_heads, L, heads_dim)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)