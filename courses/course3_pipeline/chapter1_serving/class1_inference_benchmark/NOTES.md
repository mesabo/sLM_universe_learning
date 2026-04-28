# Notes — Course 3 · ch 1 · class 1 (inference fundamentals)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | batch | single ms | batched ms | tok/s | prefill_frac | Notes |
|---|---|---|---|---|---|---|
| default | 8 |  |  |  |  |  |
|  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — sweep batch size

(Plot tok/s vs batch. Where does it flatten?)

### 2. Apply — long-context regime

(prefill_fraction with 2k-token prompt + 32 new tokens. Did it jump above 0.9? What does that imply for which serving optimization to chase?)

### 3. Stretch — vLLM comparison

(tok/s gap between vLLM and plain HF at high batch sizes. Where does the gap appear?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
