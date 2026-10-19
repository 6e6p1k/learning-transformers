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
        # with kv cache idx holds only the new tokens
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

        # kv cache: prompt runs once, then one token per step; full window -> cache rebuilt
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

        # positions come from RoPE inside the attention blocks
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

        # kv cache: same as GPT.generate (dropping old entries is only exact for 1 layer, so it's rebuilt)
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