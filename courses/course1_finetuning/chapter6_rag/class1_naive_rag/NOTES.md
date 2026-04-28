# Notes — Course 1 · ch 6 · class 1 (naive RAG)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | retriever | generator | top_k | recall@k | substring_match | mean_top_score | Notes |
|---|---|---|---|---|---|---|---|
| default | MiniLM | SmolLM2-135M-Instruct | 5 |  |  |  |  |
|  |  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — sweep top_k

(Plot recall@k and substring_match vs k. Where does recall plateau?)

### 2. Apply — swap retriever to fine-tuned encoder

(Did the ch5-trained encoder improve recall@k? Did substring_match follow?)

### 3. Stretch — add a reranker

(How much did recall@5 jump after adding the cross-encoder rerank? Cost vs benefit?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
