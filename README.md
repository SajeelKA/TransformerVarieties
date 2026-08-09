# TransformerVarieties

A from-scratch, decoder-only transformer in PyTorch where the main architectural
choices are swappable through a single config: attention type, feed-forward
activation, mixture of experts, positional embedding, and KV cache strategy. It trains a
character-level model on the TinyShakespeare dataset.

The point of the repo is comparison — train the same model with different
attention or positional embedding settings and see what changes.

## Contents

| File | Purpose |
| --- | --- |
| `TransformerMain.py` | `Transformer` and `DecoderBlock`; assembles the parts chosen by the config. Also `saveModel` / `loadModel`. |
| `AttentionTypes.py` | Multi-head, sliding-window, multi-query, and grouped-query attention. |
| `FeedForwardTypes.py` | Feed-forward block variants (ReLU, GELU, SwiGLU), with mixture of experts|
| `PositionalEmbeddingTypes.py` | Sinusoidal embeddings and RoPE. |
| `KvCacheTypes.py` | Full KV cache and a ring-buffer cache for sliding-window attention. |
| `DataLoader.py` | Downloads TinyShakespeare, builds the character vocab, yields batches. |
| `Training.py` | `Configs` dataclass, CLI argument parsing, and the training loop. |
| `Generate.py` | Greedy generation, with and without a KV cache. |
| `ParameterMethods.py` | Placeholder for initialization schemes. |
| `Demo.ipynb` | Notebook walkthrough. |
| `SavedModels/` | Pre-trained checkpoints (state dict plus the config used). |

## Setup

```bash
pip install -r requirements.txt
```

CUDA is used automatically if available; otherwise it runs on CPU.

## Training

```bash
python Training.py --attention_type gqa --pos_embed rope --d_model 64 --num_blocks 4
```

Arguments (all optional):

| Flag | Default | Values |
| --- | --- | --- |
| `--attention_type` | `gqa` | `mha`, `window`, `mqa`, `gqa` |
| `--ff` | `relu` | `relu`, `gelu`, `swiglu` |
| `--use_experts` |
| `--pos_embed` | `rope` | `rope`, `sinusoidal` |
| `--d_model` | 64 | |
| `--num_blocks` | 4 | |
| `--num_groups` | 4 | KV-head grouping for GQA |
| `--lr` | 1e-3 | |
| `--epochs` | 100 | |

Remaining settings (`batch_size`, `seq_length`, `window_size`, `num_heads`, ...)
live in the `Configs` dataclass in `Training.py`. On finishing, the model is
written to the "SavedModels" directory as `model_<timestamp>.pth` with its config
embedded, so `loadModel` can rebuild it without being told the architecture.

## Generation

```bash
python Generate.py --prompt "ROMEO:" --file_path model_gqa_RoPE.pth --gen_length 512
```

`--file_path` is resolved relative to `SavedModels/`. Generation sets
`config.inference = True`, which turns on the KV cache path in attention.

`Generate.py` also contains `generateGreedy` (no cache) and
`generateGreedyWithSlidingKV` (ring-buffer cache), which are useful for checking
that the cached path produces the same output as the uncached one.

## Notes on the implementation

- The KV cache is preallocated to `max_seq_length` and filled in place; `offset`
  is threaded through the model so attention masks and RoPE angles are sliced at
  the right positions during incremental decoding.
- Sliding-window attention uses a ring buffer of size `window_size`, so the
  masking during generation has to account for slot positions rather than
  absolute positions.
- With sinusoidal embeddings, generation breaks down past `seq_length`; RoPE is
  the option to use for longer sequences.
- `flash_attention` and `paged_attention` appear in the argument dictionary but
  are not implemented. `mlp_with_swiglu` currently uses GELU.
