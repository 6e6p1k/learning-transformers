# learning-transformers

A decoder-only transformer (GPT) written from scratch in PyTorch, trained at character level on TinyShakespeare, with Rotary Position Embeddings and a KV cache for generation. A learning repo: the experiments are small and the numbers below are reported as they were recorded, single run each.

## What is implemented

| File | Contents |
|---|---|
| `src/models/attention_block.py` | `AttentionBlock`, `AttentionBlockWithRoPE`: causal multi-head self-attention with explicit `W_q`, `W_k`, `W_v`, `W_o`, scaling, `triu` mask; optional KV cache |
| `src/models/transformer_block.py` | Post-LN block: `x = LN(x + attn(x))`, `x = LN(x + ffn(x))`, GELU FFN |
| `src/models/layer_norm.py` | LayerNorm written by hand (reference only; the blocks use `nn.LayerNorm`) |
| `src/models/RoPE.py` | RoPE from the RoFormer formula, interleaved pairs, applied to Q and K before the head split |
| `src/models/GPT_architecture.py` | `GPT` (learned absolute positions) and `GPT_RoPE`, each with `generate()` |
| `src/models/kv_cache.py` | `KVCache`: per-layer K and V for incremental decoding |
| `tests/test_kv_cache.py` | 4 tests: cached and uncached inference must agree (CPU, seconds) |
| `notebooks/testing.ipynb` | The two training runs below |
| `notebooks/first_transformer.ipynb` | An earlier 5,000-step run of the learned-position model |
| `notebooks/main.ipynb` | Notes: attention derivation, FFN as key-value memory, residual stream, RoPE motivation, linear attention |
| `notebooks/reviews.ipynb`, `notebooks/reviews/train_reviews_ru.py` | Side project: a bidirectional encoder classifier (no causal mask, CLS token) on Russian car reviews, and a rubert-tiny2 fine-tune with the HF Trainer |

## Training setup and recorded numbers

Character-level TinyShakespeare (`notebooks/shakespeare.txt`, 1,115,394 characters, 65-symbol vocab, 90/10 split). Both runs: 4 layers, d=128, 4 heads, FFN 512, context 64, batch 64, AdamW at 3e-4, 10,000 steps planned, validation every 500 steps on 50 random batches. Batches are random windows (`torch.randint`), not a sequential loader. Run on an Apple GPU (`mps`). The numbers come from the saved outputs of `notebooks/testing.ipynb`; they were not re-run for this README.

| Validation loss | Params | step 0 | 500 | 1,000 | 2,000 | 3,500 | end |
|---|---|---|---|---|---|---|---|
| `GPT`, learned positions | 818,241 | 4.385 | 2.176 | 1.974 | 1.804 | 1.689 | 1.564 at step 9,500 |
| `GPT_RoPE` | 810,049 | 4.309 | 1.985 | 1.861 | 1.745 | 1.656 | interrupted at step 3,500 |

What this does and does not show:

- 0.8M parameters, validation loss from 4.39 to about 1.56 over 10,000 steps. Samples are noisy pseudo-Shakespeare.
- RoPE is ahead of learned positions at every checkpoint up to step 3,500 (1.656 vs 1.689). This is one seed, no error bars, and the RoPE run was stopped at step 3,500, so there is no final RoPE number and no RoPE sample. It is a training-curve comparison, not a converged result.

## RoPE: what is standard and what is not

The rotation and the frequencies `theta_j = 10000^(-2(j-1)/D)` follow RoFormer, with interleaved (even, odd) pairs. Positions start at 1 instead of 0, which is a constant shift and changes nothing.

One deviation: `D` is the full embedding size (128), and the rotation is applied before the heads are split, instead of per head with `D = head_dim` (32). Each head therefore sees a quarter of the frequency range (head 0 the fastest rotations, head 3 the slowest) rather than the whole range. It is still a valid relative encoding (a query-key score depends only on the position difference, which `tests/test_kv_cache.py` relies on), but it is not the per-head RoPE of the paper. The fix is a few lines (frequencies over `heads_dim`, rotation after the `view`) and would need a retrain.

## Linear attention: notes, no code

`notebooks/main.ipynb` (the Linear Attention sections) derives the kernelised form `sim(q, k) ~ phi(q)^T phi(k)`, the `S = sum phi(k_n) v_n^T` and `z = sum phi(k_n)` aggregates that make it O(N), and why additive positional encodings break the factorisation: the position-times-content cross terms in `q_m^T k_n` need both tokens at once. RoPE, which rotates q and k separately, does not have this problem. Nothing here is implemented.

## KV cache

`generate(idx, max_new_tokens, use_cache=True)` runs the prompt once, then feeds one token per step. Each attention block appends that step's K and V to a `KVCache` (one `(B, heads, L, head_dim)` pair per layer) and attends over everything cached; the causal mask for a step with `past` cached positions is `triu(ones(L_new, past + L_new), diagonal=1 + past)`, which reduces to the usual mask when `past = 0`. In the RoPE model the new token is rotated with `offset = past`, and keys are cached after rotation, so each key is rotated once at its own position. The default `use_cache=False` path and the training path are unchanged.

When the sequence outgrows `context_len`, both models rebuild the cache from the last `context_len` tokens, which is exactly the window the uncached path recomputes, so `use_cache=True` gives the same logits as the default path at every step. The cache therefore saves work only while the sequence fits in the context.

Dropping the oldest cache entries instead (`KVCache.crop`, keeping the position counter) is exact for a one-layer RoPE model, because RoPE scores depend only on relative offsets. With two or more layers it is only an approximation: the cached keys and values of the remaining tokens were computed from hidden states that still saw the dropped tokens. A test with a one-layer model covers the exact case; `generate()` does not use the sliding variant.

`tests/test_kv_cache.py` checks, on a tiny seeded model on CPU: the RoPE `offset` against the full sequence; prefill plus one-token steps against one full forward (both models, `atol=1e-5`); the one-layer sliding RoPE cache against a recomputed window; and that `generate()` samples the same tokens with and without the cache from the same RNG state, past `context_len` as well. All 4 pass (CPU, PyTorch installed in a fresh virtual environment).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/testing.ipynb  # training: run the GPT cell or the GPT_RoPE cell
python tests/test_kv_cache.py             # from the repo root, prints "4 tests passed"
```

```python
model.generate(ctx, max_new_tokens=500)                  # original path, recomputes the window
model.generate(ctx, max_new_tokens=500, use_cache=True)  # KV cache
```

## Limitations

- One seed per experiment, no error bars, and the RoPE run is unfinished.
- The RoPE frequency layout deviates from the paper (above). No relative-position check was done on trained weights, only the numerical one in the test on random weights.
- The hand-written LayerNorm is not used by the blocks.
- Linear attention exists only as notes.
- The Russian review texts in `notebooks/reviews/` have no stated provenance or licence.
- The KV cache concatenates per step instead of writing into a preallocated buffer, and it gives no speed-up once the sequence is longer than `context_len`. No timings were measured.
- No checkpoints are stored in the repo.

## What I would do next

- Finish the RoPE run and repeat both runs over three seeds, so the comparison has error bars.
- Move RoPE to per-head frequencies and retrain; check the relative-position property on the trained model, not only on random weights.
- Preallocate the cache and measure tokens per second with and without it; then grouped-query attention on top of it (fewer K/V heads, same cache code).
- Implement the linear attention from the notes and compare it with softmax attention on a short sequence.
- Run the test in CI.
