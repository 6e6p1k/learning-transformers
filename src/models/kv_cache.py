import torch


class KVCache:
    """Keys and values of the tokens already processed, one (K, V) pair per transformer block.

    Used by ``generate(..., use_cache=True)``: the prompt is run once, then every new token is fed
    on its own and attends over the cached keys and values instead of recomputing the whole window.

    K and V are stored after the head split (and, in the RoPE model, after the rotation), so each
    entry has shape (B, num_heads, L_cached, heads_dim).
    """

    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers
        # tokens ever fed through each layer. Unlike seq_len() it keeps counting after crop(),
        # which is what RoPE needs to give the next token its true absolute position.
        self.n_seen = [0] * num_layers

    def seq_len(self, layer):
        """Number of positions currently cached for this layer."""
        return 0 if self.k[layer] is None else self.k[layer].shape[2]

    def position(self, layer):
        """Absolute position of the next token this layer will see."""
        return self.n_seen[layer]

    def update(self, layer, k_new, v_new):
        """Append this step's K and V, (B, num_heads, L_new, heads_dim); return the full cached K and V."""
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        self.n_seen[layer] += k_new.shape[2]
        return self.k[layer], self.v[layer]

    def crop(self, max_len):
        """Keep only the newest ``max_len`` positions in every layer. Positions keep counting."""
        for i in range(len(self.k)):
            if self.k[i] is None:
                continue
            if max_len <= 0:
                self.k[i], self.v[i] = None, None
            elif self.k[i].shape[2] > max_len:
                self.k[i] = self.k[i][:, :, -max_len:]
                self.v[i] = self.v[i][:, :, -max_len:]

    def reset(self):
        """Forget everything, positions included."""
        for i in range(len(self.k)):
            self.k[i], self.v[i], self.n_seen[i] = None, None, 0
