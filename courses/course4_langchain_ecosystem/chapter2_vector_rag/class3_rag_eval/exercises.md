# Exercises — Class 3: RAG Evaluation with RAGAS

## Warm-up: Add faithfulness metric

Add `Faithfulness()` to the list of RAGAS metrics in `train.py`. Run evaluation on 3 samples. Compare the faithfulness score to the semantic similarity score. Does a high faithfulness score always accompany a high semantic similarity? Explain why or why not.

## Apply: Build an adversarial eval set

Create 3 "hard" evaluation samples where the correct answer is not in the top-3 retrieved documents (e.g., by using ambiguous queries). Measure context recall on these hard samples vs the easy samples from the class corpus. Report the gap and propose one retrieval change that would improve recall on the hard set.

## Stretch: Sweep k and report

Run the full RAG evaluation for `k` in `[1, 3, 5]`. Record `context_recall` and `semantic_similarity` for each. Plot (or tabulate) how both metrics change as `k` increases. At what `k` do you get diminishing returns? Discuss the cost-quality tradeoff.
