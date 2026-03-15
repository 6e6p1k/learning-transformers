import torch
from torch import nn
from IPython.display import SVG

SVG(filename="/diagrams/rope_positional_encoding.svg")



class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def forwad(self, x):
