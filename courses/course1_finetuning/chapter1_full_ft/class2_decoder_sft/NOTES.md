# Notes — Course 1 · ch 1 · class 2 (decoder SFT with TRL)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Backbone | mode | train_loss_final | eval_loss | Notes |
|---|---|---|---|---|---|
| default | HuggingFaceTB/SmolLM2-135M-Instruct | smoke |  |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — generate before & after

(What changed in the model's output on the same prompt before and after SFT? What stayed the same?)

### 2. Apply — bigger backbone

(eval_loss and GPU memory difference between SmolLM2-135M and SmolLM2-360M.)

### 3. Stretch — packing

(Wall-clock training time with `packing=true` vs `false`. When does packing hurt?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
