# Course 1 · Chapter 6 · Class 1 — Naive RAG: encoder retrieval + decoder generation

> Goal: build the simplest end-to-end RAG pipeline against `rag-datasets/rag-mini-bioasq` — embed the corpus with one of the encoder backbones, retrieve top-k by cosine similarity, format a prompt with the retrieved passages, generate an answer with SmolLM2-Instruct, and measure both retrieval recall and answer substring match.

---

## Psycho — the mental model

Fine-tuning teaches the model new behavior. RAG teaches it new **facts** at inference time, by stuffing relevant passages into the prompt. Most production "AI" features are RAG plus a thin generation step, not custom-trained models.

The two halves of RAG fail differently:

- **Bad retrieval, good generation** → confidently wrong answers grounded on irrelevant passages.
- **Good retrieval, bad generation** → correct passages pasted with garbled or evasive prose.

A useful eval has to measure both:

- `retrieval_recall_at_k` — fraction of questions where any relevant passage made it into the top-k. Pure retriever metric, decoder-blind.
- `answer_substring_match` — fraction of generated answers that contain the reference answer text (exact substring, case-insensitive).

This class measures both on a small QA dataset so you see which half of your stack is the bottleneck.

## Academic — what's happening

Given a question $q$, a corpus $\{d_i\}$, and an encoder $f$, naive RAG computes:

$$
\mathrm{top}_k(q) = \mathrm{argtop}_k\!\left[\, f(q)^\top f(d_i) \,\right]_i
$$

then prompts the decoder with the concatenation of those passages plus the question, and asks it to answer. The standard upgrades — query rewriting, reranking, chunking, multi-hop — all preserve this skeleton; they just substitute a different `topₖ` function or post-process its output.

References:
- [Lewis et al., *Retrieval-Augmented Generation* (NeurIPS 2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper
- [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) — small biomedical QA + corpus, two configs: `question-to-passages` (queries) and `text-corpus` (passages). Parquet.
- [SmolLM2-Instruct chat template](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) — used to format the user/system messages
- [HF — `SentenceTransformer.encode`](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode)

## Engineering — what the code does

[`train.py`](./train.py) (it's an inference pipeline; "train" is preserved for class-folder uniformity):

1. Loads the encoder via `shared.backbones.load_backbone(retriever)` and the decoder via `load_backbone(generator)`.
2. Loads `text-corpus` and embeds every passage once with the encoder; caps to `corpus_n` rows for smoke speed.
3. Loads `question-to-passages`, samples `eval_n` questions.
4. For each question:
   - Embeds the question, computes top-k via cosine similarity.
   - Records whether any passage in `relevant_passage_ids` made it into top-k.
   - Renders a chat prompt: system + "Answer using these passages" + the top-k passages + the question.
   - Calls `model.generate` with `max_new_tokens` from config.
   - Records substring match against the reference answer (case-insensitive).
5. Persists a result JSON via `shared.eval_harness.run_eval`.

### Gotchas
- **Retrieval is the bottleneck for tiny encoders on technical corpora.** Pre-FT MiniLM gets ~30–60% recall@5 on BioASQ; the answer match ceiling is then capped at that. Try BGE-small or fine-tune via chapter 5 first.
- **Prompt length matters.** Each passage is a few hundred tokens; top-5 means a 1.5–2.5 k context. SmolLM2-135M has 8k context, so ample headroom — but `max_new_tokens` should be modest (64 or 128) for quick smoke.
- **Substring match is a *floor* metric.** It misses paraphrases. Real systems use exact match + F1 + LLM-as-judge. We use substring for simplicity and so the lesson is reproducible without an external grader.
- **Corpus embedding is the slow step.** With `corpus_n=2000` it's ~10–20s on a single GPU; budget for it. We don't cache between runs (the cache layer would be a chapter 6.x extension).

## Research — open questions / extensions

- Plug in the **chapter 5 fine-tuned encoder** as the retriever. Does `retrieval_recall_at_5` jump?
- Compare decoder backbones: SmolLM2-135M vs SmolLM2-360M for the generator. Substring match should improve with the bigger model, but only if retrieval was good enough to give it material to work with.
- Add a **reranker** step: retrieve top-50 with a cheap encoder, rerank with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`), keep top-5. How much recall@5 improves vs the cheap-only baseline.
- Replace substring match with **LLM-as-judge** (a third model that scores correctness). Does the ranking of retrievers change?

---

## How to run

```bash
bash courses/course1_finetuning/chapter6_rag/class1_naive_rag/run.sh
```

Smoke mode by default — embeds a 2k-passage corpus and answers 32 questions in ~1–2 min on a single GPU. First run downloads the dataset.

## How to verify

`results/full/<retriever>/course1_finetuning/chapter6_rag_class1_naive_rag/bioasq/rag-k<K>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `retrieval_recall_at_k` | 0.10 | 1.0 | Sanity: at least some questions found a relevant passage |
| `answer_substring_match` | 0.0 | 1.0 | Permissive — small models on biomedical QA struggle |
| `n_questions_evaluated` | 1 | 10000 | At least one question ran end-to-end |
| `mean_top_passage_score` | 0.0 | 2.0 | Mean cosine sim of the #1 retrieved passage |

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
