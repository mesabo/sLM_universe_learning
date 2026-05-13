# Course 1 · Chapter 5 · Class 1 — Contrastive fine-tuning of an encoder for retrieval

> **Goal:** Take a pretrained sentence-encoder (MiniLM, BGE, or GTE) and fine-tune it using **Contrastive Learning**. You will teach the model to pull "similar" sentences closer together and push "different" sentences further apart in vector space.

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are performing **Contrastive Fine-Tuning**.
*   **The Data:** We use "pairs" of sentences. In this class, we use the **Quora Duplicates** dataset (pairs of questions that mean the same thing).
*   **The Goal:** We want to optimize the model so that for any "Anchor" question, its "Positive" match has a higher similarity score than any other question in the dataset.
*   **The Loss:** We use **MultipleNegativesRankingLoss (MNRL)**, which is the industry standard for training retrievers.

### WHY are we doing this?
*   **Domain Alignment:** A general-purpose model might know that "Apple" is a fruit, but it might not know that in your specific technical support dataset, "Apple" and "MacBook" should be very close together.
*   **Better Retrieval (RAG):** Contrastive fine-tuning is the single most important step for making a RAG system accurate. It ensures the "Retriever" actually finds the right documents.
*   **Calibration:** It teaches the model a stricter "sense of distance," helping it distinguish between true paraphrases and sentences that just share a few common words.

### WHEN should you use this?
*   Use it when your RAG system is fetching the wrong documents.
*   Use it when you have a dataset of "Search Query → Relevant Document" pairs.
*   Use it when you need a custom embedding model for a specific industry (Legal, Medical, Finance).

### WHERE do the "Negatives" come from?
This is the "magic" of this class: **In-Batch Negatives**.
1.  In a batch of 64 pairs, we have 64 Anchors and 64 Positives.
2.  For Anchor #1, Positive #1 is the "Target."
3.  **The Negatives:** We treat all the *other* 63 Positives in that same batch as "Negatives."
4.  This means the model has to learn to pick the one right match out of a lineup of 64 possibilities. No manual labeling of "bad matches" is required!

### HOW does it work (The Pipeline)?
1.  **Sentence-Transformers:** We use the `sentence-transformers` library, which simplifies the training of Siamese Networks (two identical models comparing inputs).
2.  **Cosine Similarity:** The model calculates the "angle" between the Anchor and all Positives in the batch.
3.  **Ranking:** The model is penalized if the "True Positive" isn't the highest-scoring match.
4.  **Evaluation:** We use **MRR (Mean Reciprocal Rank)** to see how high the true match ranks on average. An MRR of 1.0 is a perfect score.

---

## 🧠 Psycho — the mental model

A pretrained sentence-encoder already produces semantically meaningful vectors — but it was trained on whatever distribution its authors chose. For your retrieval task (queries vs. docs in your domain), it almost certainly *under-clusters* near-duplicates and *over-separates* paraphrases. Contrastive fine-tuning fixes this by showing the model **pairs that should be close**.

The standard cheap trick is **in-batch negatives**: in a batch of `N` `(anchor, positive)` pairs, treat the other `N-1` positives as negatives for each anchor. No need to mine hard negatives explicitly — every batch produces `N²` similarity comparisons.

You don't need a generation model, you don't need preference triples — just a corpus of "these two strings should be close". For RAG (chapter 6), this is *the* knob that turns a generic encoder into a retriever your application actually trusts.

---

## 🎓 Academic — what's happening

Let $f_\theta : \text{string} \to \mathbb{R}^d$ be the encoder, normalized to the unit sphere. For a batch of pairs $\{(a_i, p_i)\}_{i=1}^N$, MultipleNegativesRankingLoss is the symmetric InfoNCE / contrastive cross-entropy:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s \cdot f_\theta(a_i)^\top f_\theta(p_i))}{\sum_{j=1}^N \exp(s \cdot f_\theta(a_i)^\top f_\theta(p_j))}
$$

**Key Terms for Students:**
*   **Anchor:** The starting sentence (e.g., a search query).
*   **Positive:** The "correct" match (e.g., the answer).
*   **In-batch Negatives:** Using other sentences in the same training batch as examples of "what NOT to find."
*   **MRR (Mean Reciprocal Rank):** A metric that tells you, on average, where the correct answer ranked (1st, 2nd, 10th?).

---

## 🛠️ Engineering — what the code does

[`train.py`](./train.py):

1.  **Model Loading:** Uses `SentenceTransformer(name)` to load the backbone.
2.  **Dataset Preparation:** Loads `quora-duplicates`. We only use the pairs where humans agreed the questions were the same.
3.  **MNRL Loss:** Sets up the `MultipleNegativesRankingLoss`.
4.  **Trainer:** Uses `SentenceTransformerTrainer`. Note that larger batches (e.g., 64 or 128) provide a much stronger learning signal than small batches.
5.  **Metrics:** Reports MRR and Recall@1. If Recall@1 is 0.80, it means the model found the perfect match 80% of the time.

### Gotchas
- **Batch Size Matters:** In contrastive learning, batch size is a hyperparameter for "difficulty." If your batch is too small, the task is too easy and the model won't learn well.
- **Eval Memory:** Calculating a similarity matrix for evaluation grows quadratically ($N^2$). Don't try to evaluate 10,000 sentences at once or you'll run out of memory!
- **Normalization:** Always ensure embeddings are normalized before calculating cosine similarity, otherwise the scores won't make sense.

---

## 🧪 Research — open questions / extensions

- Sweep `batch_size ∈ {8, 32, 64, 128}` at the same total step count. Plot MRR vs batch size. Is the curve linear in `log(batch)`?
- Train all three encoder backbones (MiniLM, BGE-small, GTE-small) to convergence on the same data. BGE-small is already retrieval-tuned — does it benefit less from your task-specific data than MiniLM?
- Add **hard negatives**: for each anchor, mine the top-1 false positive from the *current* model's embeddings every K steps and pass them explicitly via `MultipleNegativesRankingLoss` (it accepts a `negative` column too). How much does MRR jump?
- The encoder produced here is exactly what Course 1 chapter 6 (RAG) will use as the retriever. Plug it in there and compare end-to-end answer accuracy vs the un-tuned baseline.

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter5_embedding_ft/class1_contrastive/run.sh
```

Smoke mode by default (~2048 train pairs, ~50 steps, ~1–2 min on a single GPU).

## ✅ How to verify

`results/full/<backbone>/course1_finetuning/chapter5_embedding_ft_class1_contrastive/quora/mnrl-b<BATCH>.json`. Expected band (smoke):

| Metric | Passing Range | Meaning |
|---|---|---|
| `mrr` | 0.40 - 1.0 | Mean reciprocal rank (1.0 is perfect) |
| `recall_at_1` | 0.30 - 1.0 | Did the #1 result match? |
| `recall_at_5` | 0.55 - 1.0 | Was the match in the Top 5? |
| `loss_decreased` | 1 (True) | Sanity check |
