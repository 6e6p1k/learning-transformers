import torch


class KVCache:
    # K, V per layer, (B, num_heads, L, heads_dim), stored after the head split (and after RoPE)

    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers
        # tokens seen so far, keeps counting after crop() (RoPE needs the real position)
        self.n_seen = [0] * num_layers

    def seq_len(self, layer):
        # positions currently cached
        return 0 if self.k[layer] is None else self.k[layer].shape[2]

    def position(self, layer):
        # absolute position of the next token (keeps counting after crop)
        return self.n_seen[layer]

    def update(self, layer, k_new, v_new):
        # append new K, V and return the full cached ones
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        self.n_seen[layer] += k_new.shape[2]
        return self.k[layer], self.v[layer]

    def crop(self, max_len):
        # keep only the newest max_len positions
        for i in range(len(self.k)):
            if self.k[i] is None:
                continue
            if max_len <= 0:
                self.k[i], self.v[i] = None, None
            elif self.k[i].shape[2] > max_len:
                self.k[i] = self.k[i][:, :, -max_len:]
                self.v[i] = self.v[i][:, :, -max_len:]

    def reset(self):
        for i in range(len(self.k)):
            self.k[i], self.v[i], self.n_seen[i] = None, None, 0
