# cached and uncached inference should agree
# run from the repo root: python tests/test_kv_cache.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn

from src.models.GPT_architecture import GPT, GPT_RoPE
from src.models.RoPE import RoPE
from src.models.kv_cache import KVCache

VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, CONTEXT_LEN, NUM_LAYERS = 11, 16, 4, 32, 8, 2
BATCH = 2


def make_model(kind, seed=0):
    torch.manual_seed(seed)
    if kind == "learned":
        model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, CONTEXT_LEN, NUM_LAYERS,
                    loss_fn=nn.CrossEntropyLoss(),
                    positional_encoding=nn.Embedding(CONTEXT_LEN, EMBEDDING_DIM))
    else:
        model = GPT_RoPE(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, CONTEXT_LEN, NUM_LAYERS,
                         loss_fn=nn.CrossEntropyLoss())
    return model.eval()


def random_tokens(length, seed=1):
    torch.manual_seed(seed)
    return torch.randint(VOCAB_SIZE, (BATCH, length))


@torch.no_grad()
def incremental_logits(model, seq, prefill):
    # prefill tokens in one go, then one token per step; logits for every position
    cache = KVCache(len(model.blocks))
    logits, _ = model(seq[:, :prefill], cache=cache)
    chunks = [logits]
    for t in range(prefill, seq.shape[1]):
        logits, _ = model(seq[:, t:t + 1], cache=cache)
        chunks.append(logits)
    return torch.cat(chunks, dim=1)


def test_rope_offset_matches_full_sequence():
    # rotating one token with offset t == rotating it at position t of the full sequence
    rope = RoPE()
    torch.manual_seed(2)
    x = torch.randn(BATCH, CONTEXT_LEN, EMBEDDING_DIM)
    full = rope(x)
    for t in range(CONTEXT_LEN):
        single = rope(x[:, t:t + 1], offset=t)
        assert torch.allclose(full[:, t:t + 1], single, atol=1e-6), f"position {t}"


def test_cached_matches_full_forward_within_context():
    # prefill + one-token steps == one full forward, up to context_len
    seq = random_tokens(CONTEXT_LEN)
    for kind in ("learned", "rope"):
        model = make_model(kind)
        with torch.no_grad():
            full, _ = model(seq)
        inc = incremental_logits(model, seq, prefill=3)
        assert full.shape == inc.shape
        assert torch.allclose(full, inc, atol=1e-5, rtol=1e-5), \
            f"{kind}: max abs diff {(full - inc).abs().max().item()}"


def test_rope_sliding_cache_matches_recomputed_window_single_layer():
    # dropping the oldest entries is exact for 1 layer only (deeper layers saw the dropped tokens)
    torch.manual_seed(0)
    model = GPT_RoPE(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, CONTEXT_LEN, 1,
                     loss_fn=nn.CrossEntropyLoss()).eval()
    seq = random_tokens(CONTEXT_LEN + 12)
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        for t in range(seq.shape[1]):
            if cache.seq_len(0) + 1 > CONTEXT_LEN:
                cache.crop(CONTEXT_LEN - 1)
            cached, _ = model(seq[:, t:t + 1], cache=cache)
            window = seq[:, max(0, t + 1 - CONTEXT_LEN):t + 1]
            reference, _ = model(window)
            diff = (cached[:, -1, :] - reference[:, -1, :]).abs().max().item()
            assert torch.allclose(cached[:, -1, :], reference[:, -1, :], atol=1e-4, rtol=1e-4), \
                f"step {t}: max abs diff {diff}"
    assert cache.seq_len(0) == CONTEXT_LEN
    assert cache.position(0) == seq.shape[1]


def test_generate_same_tokens_with_and_without_cache():
    # same rng state -> same tokens with and without cache, also past context_len
    prompt = random_tokens(3)
    for kind in ("learned", "rope"):
        model = make_model(kind)
        torch.manual_seed(123)
        plain = model.generate(prompt, max_new_tokens=3 * CONTEXT_LEN)
        torch.manual_seed(123)
        cached = model.generate(prompt, max_new_tokens=3 * CONTEXT_LEN, use_cache=True)
        assert plain.shape == (BATCH, 3 + 3 * CONTEXT_LEN)
        assert torch.equal(plain, cached), f"{kind}: cached and uncached generate() diverged"


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(tests)} tests passed")
