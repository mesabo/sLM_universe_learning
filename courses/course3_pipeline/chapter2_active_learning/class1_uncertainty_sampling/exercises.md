# Exercises — Course 3 · ch 2 · class 1 (active learning)

## 1. Warm-up — sweep the round count

```bash
for r in 2 4 8; do
  bash run.sh --config configs/default.yaml active.n_rounds=$r
done
```

Plot `delta_accuracy` vs `n_rounds`. Does the gap between uncertainty and random grow, plateau, or shrink as you give them more labels?

## 2. Apply — margin sampling

Add a third strategy: **margin sampling** — pick examples where `p_top1 - p_top2` is *smallest* (the model is most confused between two classes). Implement `_query_margin` in `train.py`. Does it beat plain max-entropy on AG News?

## 3. Stretch — diversity-aware querying

High-entropy examples are often near-duplicates ("this article is hard for the same reason that one is"). Cluster the top-100 high-entropy examples (e.g. KMeans on encoder embeddings, k=`query_size`), pick one per cluster. Does this hybrid beat raw uncertainty when the unlabeled pool has duplicates?
