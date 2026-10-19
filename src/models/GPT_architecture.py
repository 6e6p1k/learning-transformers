import torch
from torch import nn

from src.models.transformer_block import TransformerBlock, TransformerBlockWithRoPE
from src.models.kv_cache import KVCache


class GPT(nn.Module):
    def __init__(self,
                vocab_size,
                embedding_dim,
                num_heads,
                hidden_dim,
                context_len,
                num_transformer_blocks,
                loss_fn,
                positional_encoding
        ):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size

        self.loss = loss_fn
        self.pos_emb = positional_encoding

        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_transformer_blocks)
        ])
        self.ln_f  = nn.LayerNorm(embedding_dim)
        self.head  = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None, cache=None):
        B, L = idx.shape
        # with a KV cache, idx holds only the new tokens: they continue from the last cached position
        past = 0
        if cache is not None:
            past = cache.position(0)
            assert past + L <= self.context_len, "no learned position embedding beyond context_len"
        positions = torch.arange(past, past + L, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(positions)  # (B, L, D)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=False):
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_crop = idx[:, -self.context_len:]
                logits, _ = self(idx_crop)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_tok], dim=1)
            return idx

        # KV cache: the prompt is run once, after that every step feeds a single token and attends
        # over the cached keys and values. The learned positional embedding is absolute, so the cache
        # is only valid while the window does not slide: once the sequence outgrows context_len the
        # cache is rebuilt from the last context_len tokens, which is exactly the window the uncached
        # path recomputes. Both paths therefore give the same logits at every step.
        cache = KVCache(len(self.blocks))
        tokens = idx[:, -self.context_len:]
        for _ in range(max_new_tokens):
            logits, _ = self(tokens, cache=cache)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if cache.seq_len(0) + 1 > self.context_len:
                cache.reset()
                tokens = idx[:, -self.context_len:]
            else:
                tokens = next_tok
        return idx
    


class GPT_RoPE(nn.Module):
    def __init__(self,
                vocab_size,
                embedding_dim,
                num_heads,
                hidden_dim,
                context_len,
                num_transformer_blocks,
                loss_fn,
        ):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        
        self.loss = loss_fn

        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.blocks  = nn.Sequential(*[
            TransformerBlockWithRoPE(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_transformer_blocks)
        ])
        self.ln_f  = nn.LayerNorm(embedding_dim)
        self.head  = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None, cache=None):
        B, L = idx.shape
        positions = torch.arange(L, device=idx.device)

        # positions are applied inside the attention blocks (RoPE); with a KV cache each block
        # reads the offset of the new tokens from the cache
        x = self.tok_emb(idx)  # (B, L, D)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=False):
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_crop = idx[:, -self.context_len:]
                logits, _ = self(idx_crop)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_tok], dim=1)
            return idx

        # KV cache: the prompt is run once, after that every step feeds a single token and attends
        # over the cached keys and values. Once the sequence outgrows context_len the cache is rebuilt
        # from the last context_len tokens, exactly as in GPT.generate: simply dropping the oldest
        # entries (KVCache.crop) would be exact for one layer only, because in deeper layers the cached
        # keys and values of the remaining tokens were computed from hidden states that still saw the
        # dropped tokens, whereas the uncached path recomputes the window from scratch.
        cache = KVCache(len(self.blocks))
        tokens = idx[:, -self.context_len:]
        for _ in range(max_new_tokens):
            logits, _ = self(tokens, cache=cache)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if cache.seq_len(0) + 1 > self.context_len:
                cache.reset()
                tokens = idx[:, -self.context_len:]
            else:
                tokens = next_tok
        return idx