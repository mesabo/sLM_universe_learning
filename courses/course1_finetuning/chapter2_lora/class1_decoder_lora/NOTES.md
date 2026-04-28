# Notes — Course 1 · ch 2 · class 1 (LoRA on decoder)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | r | alpha | trainable_ratio_pct | eval_loss | Notes |
|---|---|---|---|---|---|
| default | 16 | 32 |  |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — read trainable params print-out

(Verify the math: 4 target_modules × 2 × r × hidden ≈ trainable params per layer.)

### 2. Apply — sweep r

(eval_loss and adapter file size for r ∈ {4, 16, 64}. At what rank does the adapter stop being "tiny"?)

### 3. Stretch — multi-adapter loading

(Two adapters from different SFT runs loaded onto the same base; switching between them via `model.set_adapter`. How does this set up Course 2 ch4?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
