# Notes — Course 1 · ch 5 · class 1 (contrastive embedding FT)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | backbone | batch | pre_mrr | post_mrr | delta_mrr | recall_at_1 | Notes |
|---|---|---|---|---|---|---|---|
| default | MiniLM | 64 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — sweep batch size

(Plot MRR vs log(batch). Linear?)

### 2. Apply — compare backbones

(`delta_mrr` per backbone. Did BGE-small benefit less than MiniLM? Why?)

### 3. Stretch — hard-negative mining

(How did you implement the mining loop? Did MRR jump materially?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
