# Exercises — Course 1 · ch 6 · class 1 (naive RAG)

## 1. Warm-up — sweep `top_k`

```bash
for k in 1 3 5 10; do
  bash run.sh --config configs/default.yaml retrieval.top_k=$k
done
```

How does `retrieval_recall_at_k` change vs `answer_substring_match`? Recall should rise monotonically with `k` (more passages = more chance the right one is in there) but match might saturate — at some point more passages just dilute the prompt.

## 2. Apply — swap retriever to your fine-tuned encoder

After running Course 1 ch5 (contrastive embedding FT), point this class at the saved sentence-transformers checkpoint:

```bash
bash run.sh --config configs/default.yaml backbone=<path/to/saved/st-checkpoint>
```

Did `retrieval_recall_at_k` improve? Did `answer_substring_match` follow?

## 3. Stretch — add a reranker

Retrieve top-50 with the cheap MiniLM encoder, then rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep top-5. (Modify `train.py` between the topk and prompt-building steps.) How much does recall@5 jump? At what corpus size does the reranker stop being affordable?
