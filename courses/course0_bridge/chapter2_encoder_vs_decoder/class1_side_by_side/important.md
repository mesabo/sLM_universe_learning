# Encoder vs Decoder — Key Points (Important)

Short answer
-----------
- Encoders are optimized to produce a single-shot semantic vector (fast, calibrated for retrieval).
- Decoders are optimized to predict next tokens autoregressively (designed for generation).

Why they differ
----------------
- **Purpose:** Encoder → map text → vector(s). Decoder → extend text (generate). Different training objectives.
- **Attention / Masking:** Encoder uses full (bidirectional) attention; decoder uses causal (autoregressive) attention.
- **Output head:** Encoder returns pooled hidden states; decoder has an LM head (hidden → vocab logits) for generation.
- **Training & Calibration:** Encoders often trained with contrastive/MLM losses so vector distances are meaningful; decoders are trained for token prediction, not calibrated similarity.
- **Runtime & cost:** Encoder = one forward → cheap. Decoder generation = autoregressive loop + KV cache → higher latency and memory.

Concrete example
----------------
- Encoder run: `v = encoder.encode(text)` → one forward, vector `v` (dim = hidden_size); use `v` for nearest-neighbor retrieval.
- Decoder-as-encoder (pooling): `h = decoder.forward(text, return_hidden=True); v' = pool(h)` → may work, but:
  - `v'` was not trained as a calibrated embedding;
  - you pay decoder compute and memory; it's slower and larger.

Mini code (pseudo)
-------------------
Encoder:

```py
# sentence-encoder
v = encoder(text)
# use v for retrieval: nn.search(v)
```

Decoder (pooling hidden states):

```py
# decoder without generation
h = decoder.forward(text, return_hidden=True)
v = h.mean(dim=1)  # or pool with CLS / pooling function
```

Analogy
-------
- Encoder = scanner/summary machine: reads a page and returns a compact, comparable summary.
- Decoder = storyteller: trained to continue the story line-by-line; not primarily tuned to compress into a comparable summary.

When to combine
----------------
- RAG = encoder (retrieve) + decoder (generate) — standard pattern. Use each model for the job it was trained for.

If you'd like I can add a minimal verification snippet or a short diagram into the class README instead of this file. Tell me which you prefer.

---
### Runnable verifier (Python)
Copy and run this locally to verify every step and see the numeric output.

```python
import numpy as np

# X (2x6)
X = np.array([[1,0,1,0,0,0],
        [0,1,0,1,1,1]])

# W_Q (6x6) as in the example (selector-style)
WQ = np.array([
  [1,0,0,0,0,0],
  [0,1,0,0,0,0],
  [0,0,1,0,0,0],
  [0,0,0,1,0,0],
  [0,0,0,0,1,0],
  [0,0,0,0,0,1],
])

Q = X.dot(WQ)
print("X (2x6):\n", X)
print("\nW_Q (6x6):\n", WQ)
print("\nQ = X @ W_Q (2x6):\n", Q)

# reshape to heads: (L, N_att, d) with N_att=3, d=2
Q_heads = Q.reshape(2, 3, 2)
print("\nQ heads shape:", Q_heads.shape)
print("Head1:\n", Q_heads[:,0,:])
print("Head2:\n", Q_heads[:,1,:])
print("Head3:\n", Q_heads[:,2,:])

# Detailed elementwise example: compute Q[1,4] (t1,row index 1, col index 4)
row = 1; col = 4
products = X[row, :] * WQ[:, col]
print("\nDetailed dot product for Q[1,4]:")
print("X[1,:] =", X[row,:])
print("WQ[:,4] =", WQ[:,col])
print("Elementwise products:", products)
print("Sum -> Q[1,4] =", products.sum())
```

### Fully explicit elementwise example (Q[1,4])

From the matrices above: Q[1,4] = sum_{r=0..5} X[1,r] * W_Q[r,4].

- X[1,:] = [0, 1, 0, 1, 1, 1]
- W_Q[:,4] = column c4 = [0, 0, 0, 0, 1, 0]

Elementwise multiply:
- [0*0, 1*0, 0*0, 1*0, 1*1, 1*0] = [0, 0, 0, 0, 1, 0]

Sum = 1 -> Q[1,4] = 1.

This kind of explicit verifier shows every multiplication and sum so no step is left implicit.

---
## Explicit numeric matrix example (fully worked)

Setup
- Sequence length `L = 2` (two tokens)
- Hidden dim `H = 6`
- `num_attention_heads = 3`, head dim `d = 2` (so `N_att * d = 6`)

Input X (L × H)
| token | r0 | r1 | r2 | r3 | r4 | r5 |
|---:|---:|---:|---:|---:|---:|---:|
| t0 | 1 | 0 | 1 | 0 | 0 | 0 |
| t1 | 0 | 1 | 0 | 1 | 1 | 1 |

W_Q constructed as three horizontal blocks (H × 6). For clarity we choose a selector matrix so columns pick rows:
W_Q (6×6)
| row\\col | c0 | c1 | c2 | c3 | c4 | c5 |
|---:|---:|---:|---:|---:|---:|---:|
| r0 | 1 | 0 | 0 | 0 | 0 | 0 |
| r1 | 0 | 1 | 0 | 0 | 0 | 0 |
| r2 | 0 | 0 | 1 | 0 | 0 | 0 |
| r3 | 0 | 0 | 0 | 1 | 0 | 0 |
| r4 | 0 | 0 | 0 | 0 | 1 | 0 |
| r5 | 0 | 0 | 0 | 0 | 0 | 1 |

Compute Q = X · W_Q (2×6) — each Q element is a dot product of X row with W_Q column.
Q (L×6)
| token | q0 | q1 | q2 | q3 | q4 | q5 |
|---:|---:|---:|---:|---:|---:|---:|
| t0 | 1 | 0 | 1 | 0 | 0 | 0 |
| t1 | 0 | 1 | 0 | 1 | 1 | 1 |

Reshape into heads (L × N_att × d):
- Head1 = [q0,q1]
- Head2 = [q2,q3]
- Head3 = [q4,q5]

Head1 (L×d)
| token | h1_0 | h1_1 |
|---:|---:|---:|
| t0 | 1 | 0 |
| t1 | 0 | 1 |

Head2 (L×d)
| token | h2_0 | h2_1 |
|---:|---:|---:|
| t0 | 1 | 0 |
| t1 | 0 | 1 |

Head3 (L×d)
| token | h3_0 | h3_1 |
|---:|---:|---:|
| t0 | 0 | 0 |
| t1 | 1 | 1 |

Notes
- This shows how `W_Q` (H×(N_att·d)) projects hidden coordinates into per-head query vectors. Reshaping groups consecutive `d` columns into heads. Each Q entry is computed by standard matrix multiplication `Q = X @ W_Q`.
- You can craft any numeric `W_Q`; the same multiplication rules apply — the example uses a simple selector matrix so results are easy to trace.

