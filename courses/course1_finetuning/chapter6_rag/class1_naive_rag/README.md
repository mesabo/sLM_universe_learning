# Course 1 · Chapter 6 · Class 1 — Naive RAG: encoder retrieval + decoder generation

> **Goal:** Build a complete **Retrieval-Augmented Generation (RAG)** pipeline. You will combine an encoder (to find information) and a decoder (to answer questions) to create a system that can answer questions about facts it wasn't originally trained on.

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are building a **Naive RAG Pipeline**.
*   **The Component (Retriever):** An **Encoder** (like MiniLM) that searches a database of knowledge.
*   **The Component (Generator):** A **Decoder** (like SmolLM2) that reads the search results and writes a final answer.
*   **The Data:** We use **BioASQ**, a dataset of biomedical questions and scientific passages.

### WHY are we doing this?
*   **Solving Hallucinations:** Large models often make things up. RAG forces the model to base its answer on real documents provided in the prompt.
*   **Up-to-Date Knowledge:** You can update the model's knowledge simply by updating the database of documents—no retraining or fine-tuning required.
*   **Transparency:** RAG provides "citations." You can see exactly which document the model used to generate its answer.

### WHEN should you use this?
*   Use **RAG** when your model needs to answer questions about private data (e.g., your company's internal wiki).
*   Use it when facts change frequently (e.g., news or stock prices).
*   Use it for any task where "grounding" in reality is more important than creative writing.

### WHERE do the parts live?
The RAG system lives in two stages:
1.  **Storage Stage:** You turn your documents into vectors using the encoder and store them in a "Vector Database."
2.  **Inference Stage:** When a user asks a question, the system finds the closest vectors, pastes the text into the prompt, and sends it to the generator.

### HOW does it work (The Pipeline)?
1.  **Indexing:** The encoder "reads" the entire corpus of scientific passages and saves their vector summaries.
2.  **Retrieval:** When a question comes in (e.g., "What is a protein?"), the encoder finds the Top-K most similar passages in the corpus.
3.  **Augmentation:** We "stuff" those passages into a prompt template: *"Context: [Passages]... Question: [Question]... Answer:"*.
4.  **Generation:** The decoder reads this massive prompt and generates a concise answer based *only* on the context.

---

## 🧠 Psycho — the mental model

Fine-tuning teaches the model new behavior. RAG teaches it new **facts** at inference time, by stuffing relevant passages into the prompt. Most production "AI" features are RAG plus a thin generation step, not custom-trained models.

The two halves of RAG fail differently:

- **Bad retrieval, good generation** → confidently wrong answers grounded on irrelevant passages.
- **Good retrieval, bad generation** → correct passages pasted with garbled or evasive prose.

A useful eval has to measure both:

- `retrieval_recall_at_k` — fraction of questions where any relevant passage made it into the top-k. Pure retriever metric, decoder-blind.
- `answer_substring_match` — fraction of generated answers that contain the reference answer text (exact substring, case-insensitive).

---

## 🎓 Academic — what's happening

Given a question $q$, a corpus $\{d_i\}$, and an encoder $f$, naive RAG computes:

$$
\mathrm{top}_k(q) = \mathrm{argtop}_k\!\left[\, f(q)^\top f(d_i) \,\right]_i
$$

then prompts the decoder with the concatenation of those passages plus the question.

**Key Terms for Students:**
*   **Recall@K:** Did the correct document appear in the top K search results? (e.g., Recall@5 means "was it in the top 5?").
*   **Context Window:** The maximum number of words a decoder can read at once. RAG is limited by this "memory" size.
*   **Grounding:** The act of forcing a model to use provided evidence rather than its own internal memory.

---

## 🛠️ Engineering — what the code does

[`train.py`](./train.py) (Inference pipeline):

1.  **Backbone Loading:** Loads both the encoder (Retriever) and decoder (Generator).
2.  **Corpus Embedding:** The most expensive step. We turn 2,000+ medical passages into vectors.
3.  **Similarity Search:** Uses Cosine Similarity to find the best matches for each question.
4.  **Prompt Engineering:** Formats the retrieved text into a "Chat Template" that tells the model exactly what to do with the information.
5.  **Generation:** SmolLM2-135M reads the prompt and attempts to answer the medical question.

### Gotchas
- **Retrieval Bottleneck:** If your retriever fails to find the right passage, even the smartest generator will fail. RAG quality starts with the search.
- **Biomedical Complexity:** Medical terms are difficult. Tiny models like SmolLM2-135M might struggle with very complex biological logic compared to a 70B parameter model.
- **Quadratic Memory:** Embedding a massive corpus takes time and VRAM. In production, you would use a dedicated Vector DB like FAISS, Qdrant, or Pinecone.

---

## 🧪 Research — open questions / extensions

- Plug in the **chapter 5 fine-tuned encoder** as the retriever. Does `retrieval_recall_at_5` jump?
- Compare decoder backbones: SmolLM2-135M vs SmolLM2-360M for the generator. Substring match should improve with the bigger model, but only if retrieval was good enough to give it material to work with.
- Add a **reranker** step: retrieve top-50 with a cheap encoder, rerank with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`), keep top-5. How much recall@5 improves vs the cheap-only baseline.
- Replace substring match with **LLM-as-judge** (a third model that scores correctness). Does the ranking of retrievers change?

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter6_rag/class1_naive_rag/run.sh
```

Smoke mode by default — embeds a 2k-passage corpus and answers 32 questions in ~1–2 min on a single GPU. First run downloads the dataset.

## ✔ How to verify

`results/full/<retriever>/course1_finetuning/chapter6_rag_class1_naive_rag/bioasq/rag-k<K>.json`. Expected band (smoke):

| Metric | Passing Range | Meaning |
|---|---|---|
| `retrieval_recall_at_k` | 0.10 - 1.0 | Did the retriever find ANY relevant info? |
| `answer_substring_match` | 0.0 - 1.0 | Did the model answer correctly? |
| `n_questions_evaluated` | 1+ | Did the script actually finish? |
| `mean_top_passage_score` | 0.0 - 2.0 | How "confident" was the search engine? |

The retrieval metric is the headline; if it's near 0 even at `k=10`, your retriever isn't matching the corpus distribution at all (often because the encoder wasn't trained on this domain).

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections (Psycho / Academic / Engineering / Research) are present and ≥ 2 paragraphs each.
- [ ] Every reference link points at an official source (paper / HF doc / repo) where one exists.
- [ ] `train.py` and `eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written by `eval.py`.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises (warm-up / apply / stretch).
- [ ] Result JSON path matches the layout `results/full/<backbone>/<course>/<class>/<task>/<method>.json`.
- [ ] At least one smoke-mode run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
