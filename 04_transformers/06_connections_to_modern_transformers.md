# Connections to Modern Transformers

How the building blocks from Notebooks 01–05 appear in modern architectures, and what changed since "Attention Is All You Need" (2017).

---

## Architecture Variants

### Encoder-Only (BERT, 2018)

**Pre-training objective:** Masked Language Modeling (MLM)

- Randomly mask 15% of tokens, predict the masked tokens
- Uses bidirectional attention (no causal mask) — each token attends to all others
- Also: Next Sentence Prediction (NSP) — later shown to be unnecessary (RoBERTa)

**Architecture:** Stack of encoder blocks (identical to what we built in Notebook 04)

- BERT-base: 12 layers, d_model=768, 12 heads, d_ff=3072 → 110M params
- BERT-large: 24 layers, d_model=1024, 16 heads, d_ff=4096 → 340M params

**Fine-tuning:** Add a task-specific head (linear layer) on top of [CLS] token or mean pool. Fine-tune all parameters on labeled data. This is what we approximated in Notebook 05, except BERT starts with pre-trained weights from billions of tokens.

**Key insight:** Pre-training on unlabeled text gives the model rich language understanding. Fine-tuning on small labeled datasets is enough for most NLP tasks.

### Decoder-Only (GPT, 2018–present)

**Pre-training objective:** Causal Language Modeling (next token prediction)
$$P(x_t | x_1, \ldots, x_{t-1})$$

Uses the causal mask from Notebook 01 — each position can only attend to previous positions.

**Architecture:** Stack of decoder blocks (same as our encoder blocks but with causal masking)

- GPT-2: 12–48 layers, d_model=768–1600
- GPT-3: 96 layers, d_model=12288, 96 heads → 175B params
- GPT-4: details not published, estimated >1T params

**Key difference from BERT:** Unidirectional attention. Each token only sees the past. This makes it naturally suited for generation (sample next token, feed back, repeat).

### Encoder-Decoder (T5, 2019)

**Pre-training objective:** Span corruption (mask spans of text, predict them)

**Architecture:**

- Encoder: bidirectional self-attention (like BERT)
- Decoder: causal self-attention + cross-attention to encoder output
- Cross-attention uses Q from decoder, K/V from encoder — exactly the cross-attention pattern from Notebook 02

**Key insight:** "Text-to-text" framing — every NLP task becomes: input text → output text. Translation, summarization, QA, classification all use the same architecture.

---

## What Changed Since 2017

### Normalization

| Component | Original (2017) | Modern (2023+) |
|-----------|-----------------|----------------|
| Norm type | LayerNorm | RMSNorm |
| Norm placement | Post-LN | Pre-LN |

**RMSNorm** (Root Mean Square Layer Normalization):
$$\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

Drops the mean-centering step of LayerNorm. Cheaper to compute, works just as well empirically. Used in LLaMA, Mistral.

### Position Encoding

| Method | Used in | Pros | Cons |
|--------|---------|------|------|
| Sinusoidal (fixed) | Original Transformer | No parameters, clean math | Doesn't extrapolate well in practice |
| Learned absolute | BERT, GPT-2 | Flexible | Can't extrapolate beyond training length |
| RoPE (Rotary) | LLaMA, Mistral, GPT-NeoX | Relative position, extrapolates | Slightly more complex |
| ALiBi | BLOOM | No parameters, simple | Linear bias may be limiting |

**RoPE** (Rotary Position Embedding) — from Notebook 03:

- Applies rotation matrix to Q and K based on position
- Attention score q_m^T k_n depends only on relative position (m-n)
- Naturally encodes relative position without explicit bias terms
- Can be extended to longer sequences with position interpolation (YaRN, NTK-aware scaling)

### Activation Functions

| Original | Modern |
|----------|--------|
| ReLU in FFN | SwiGLU |

**SwiGLU:**
$$\text{SwiGLU}(x) = (\text{Swish}(xW_1)) \odot (xW_{\text{gate}})$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

Three weight matrices instead of two, but d_ff is reduced (typically 8/3 × d_model instead of 4×) to keep total parameters similar. Used in LLaMA, PaLM. Empirically better than ReLU across scales.

### Attention Efficiency

**Flash Attention** (2022):

- Does NOT change the math — computes exactly the same result
- Changes the memory access pattern: processes attention in blocks (tiles)
- Avoids materializing the full n×n attention matrix in HBM (high bandwidth memory)
- Instead, computes attention block-by-block using fast SRAM
- Result: 2-4x faster, uses O(n) memory instead of O(n²)
- Now standard in all major frameworks (PyTorch 2.0+)

**Grouped Query Attention (GQA):**

- Standard MHA: each head has its own Q, K, V projections
- Multi-Query Attention (MQA): single K, V shared across all heads (saves KV cache memory)
- GQA: groups of heads share K, V (compromise between MHA and MQA)
- LLaMA 2 (70B) uses GQA with 8 KV heads for 64 query heads
- Reduces KV cache memory by 8x with minimal quality loss

### Training Techniques

| Technique | Purpose |
|-----------|---------|
| Learning rate warmup | Stabilize early training (we used this in Notebook 05) |
| Cosine decay | Smooth LR reduction |
| Gradient clipping | Prevent exploding gradients |
| Weight decay (AdamW) | Regularization without interfering with Adam's moment estimates |
| Mixed precision (FP16/BF16) | 2x memory savings, faster compute |
| Gradient accumulation | Simulate larger batch sizes |
| Data parallelism | Distribute batches across GPUs |
| Tensor parallelism | Split individual layers across GPUs |
| Pipeline parallelism | Split layer stack across GPUs |

---

## Scale

| Model | Year | Params | Training tokens | Training cost (est.) |
|-------|------|--------|----------------|---------------------|
| Original Transformer | 2017 | 65M | ~1B | ~$100 |
| BERT-base | 2018 | 110M | 3.3B | ~$7K |
| GPT-2 | 2019 | 1.5B | 40B | ~$50K |
| GPT-3 | 2020 | 175B | 300B | ~$4.6M |
| LLaMA 2 | 2023 | 70B | 2T | ~$2M |
| LLaMA 3 | 2024 | 405B | 15T | ~$30M+ |

The architecture is fundamentally the same. Scale (parameters + data + compute) is the primary driver of capability improvements.

---

## Reading List

To go deeper from here:

1. **"Language Models are Few-Shot Learners" (GPT-3, Brown et al. 2020)**: scaling laws, in-context learning
2. **"Training Compute-Optimal Large Language Models" (Chinchilla, Hoffmann et al. 2022)**: optimal ratio of parameters to data
3. **"FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al. 2022)**: IO-awareness insight
4. **"LLaMA: Open and Efficient Foundation Language Models" (Touvron et al. 2023)**: modern architecture choices explained
5. **"Attention Is All You Need" revisited**
