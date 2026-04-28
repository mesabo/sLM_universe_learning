# Exercises — Course 1 · ch 5 · class 1 (contrastive embedding FT)

## 1. Warm-up — sweep batch size

```bash
for b in 8 32 64 128; do
  bash run.sh --config configs/default.yaml train.per_device_batch=$b
done
```

Plot MRR vs batch size. The InfoNCE objective gets harder (more negatives in the denominator) as batch grows — does MRR improve roughly linearly in `log(batch)`?

## 2. Apply — compare backbones

Run smoke against all three encoder backbones:

```bash
for bb in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 thenlper/gte-small; do
  bash run.sh --config configs/default.yaml backbone="$bb"
done
```

BGE-small was already retrieval-tuned by its authors. Does it benefit *less* from your task-specific training than MiniLM does? Tabulate `delta_mrr` from the result JSONs' `extras.delta_mrr`.

## 3. Stretch — hard-negative mining

Modify `_retrieval_metrics` to also return the **closest-but-wrong** anchor per item (the one your current model confuses with the true positive). Save those as a new `negative` column and pass to `MultipleNegativesRankingLoss` (it accepts an explicit negative). Re-train. Did MRR jump?
