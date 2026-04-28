# Notes — Course 1 · ch 3 · class 1 (QLoRA: 4-bit base + LoRA adapters)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | quant_type | double_quant | base_in_4bit | eval_loss | GPU MB | Notes |
|---|---|---|---|---|---|---|
| default | nf4 | true |  |  |  |  |
|  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — measure the memory win

(Per-GPU memory: ch1 class 2 (full SFT bf16) vs this class (QLoRA). Is the ratio close to the theoretical ~4×?)

### 2. Apply — turn off double quantization

(Memory and loss difference with `bnb_4bit_use_double_quant: false`. Why is double quant "almost free"?)

### 3. Stretch — QLoRA on 135M

(Wall-clock training time vs ch2 (LoRA over bf16) at the same scale. When does QLoRA's overhead become worth it?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
