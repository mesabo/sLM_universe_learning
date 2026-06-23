# Class 3 — RAG Evaluation with RAGAS

> Goal: score a RAG system with RAGAS so retrieval and generation quality stop being guesswork, then map the resulting metrics back to concrete fixes in the pipeline.

## Psycho Mode

You built a RAG pipeline. Is it good? Without measurement you are flying blind. RAGAS (RAG Assessment) provides a principled evaluation framework with metrics that diagnose specific failure modes. Context recall tells you whether the retriever is finding relevant passages. Faithfulness tells you whether the generator is hallucinating. Answer relevance tells you whether the output actually addresses the question. Together they form a diagnostic dashboard.

The mental model: think of RAGAS as a "gold standard test suite" for RAG. You provide a dataset of (question, reference answer, reference context) triples. RAGAS runs your pipeline, measures how much reference context was retrieved, and uses an LLM-as-judge to assess the quality of the generated answer. The LLM judge is itself a component you must configure — using a local sLM as judge gives different scores than using GPT-4.

## Academic Mode

Given a RAG pipeline $\Pi$ and an evaluation dataset $\mathcal{D} = \{(q_i, a_i^*, C_i^*)\}$ where $q_i$ is a question, $a_i^*$ is the reference answer, and $C_i^* = \{c_{i,1}, \ldots, c_{i,m}\}$ is the set of reference contexts:

Context Recall: fraction of reference context sentences covered by retrieved context $C_i$:

$$\text{ContextRecall} = \frac{1}{|\mathcal{D}|} \sum_i \frac{|\{c \in C_i^* : \exists r \in \Pi_{\text{ret}}(q_i), c \subseteq r\}|}{|C_i^*|}$$

Semantic Similarity: embedding cosine between generated answer $a_i = \Pi_{\text{gen}}(q_i)$ and reference $a_i^*$:

$$\text{SemanticSimilarity} = \frac{1}{|\mathcal{D}|} \sum_i \cos(\phi(a_i), \phi(a_i^*))$$

RAGAS documentation and paper: [https://docs.ragas.io/](https://docs.ragas.io/).

## Engineering Mode

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextRecall, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

rows = [{
    "user_input": question,
    "retrieved_contexts": [r.page_content for r in retrieved_docs],
    "response": generated_answer,
    "reference": reference_answer,
}]
dataset = EvaluationDataset.from_list(rows)
result = evaluate(
    dataset,
    metrics=[ContextRecall(), SemanticSimilarity()],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)
df = result.to_pandas()
print(df[["context_recall", "semantic_similarity"]].mean())
```

Gotchas:
- RAGAS metrics that call the LLM (faithfulness, answer relevance) are slow with local models. Start with `SemanticSimilarity` and `ContextRecall` for speed.
- The `EvaluationDataset.from_list` API changed across RAGAS versions; pin to `ragas>=0.2`.
- `LangchainLLMWrapper` expects a chat model (not completion). Use `HuggingFacePipeline` wrapped in `ChatHuggingFace`.

Config keys: `embed_backbone`, `backbone`, `limits.smoke.n_docs`, `limits.smoke.n_eval_samples`, `limits.smoke.k`.

## Research Mode

RAGAS metrics are LLM-evaluated, which introduces its own biases: the judge model may favor fluent-but-unfaithful answers, and small local judges may not reliably assess faithfulness. Research directions: (1) human correlation studies — how well do RAGAS scores correlate with human judgments? (2) adversarial datasets — can you construct questions where context recall is high but faithfulness is low? (3) metric sensitivity — how much do RAGAS scores change across retriever k values, embedding models, and LLM temperatures?

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class3_rag_eval/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class3_rag_eval/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `context_recall` | [0.3, 1.0] |
| `answer_similarity` | [0.5, 1.0] |
| `eval_complete` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
