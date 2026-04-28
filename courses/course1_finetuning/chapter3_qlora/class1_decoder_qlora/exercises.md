# Exercises — Course 1 · ch3 · class 1

## 1. Warm-up — measure the memory win

Before and after switching to QLoRA, run `nvidia-smi --query-gpu=memory.used --format=csv` while the model is loaded. What's the per-GPU memory footprint of:
- ch1 class 2: full SFT of SmolLM2-360M in bf16
- this class: QLoRA on SmolLM2-360M

How does the ratio compare to the theoretical ~4× memory reduction?

## 2. Apply — turn off double quantization

Set `quantization.bnb_4bit_use_double_quant: false`. Re-run the smoke. Memory difference? Loss difference? Why is double quant "almost free"?

## 3. Stretch — QLoRA on the 135M model

Run with `backbone=HuggingFaceTB/SmolLM2-135M-Instruct`. Compare wall-clock training time to ch2 (LoRA over bf16). Is QLoRA *worth it* at this scale, or does it just add overhead? When does the trade-off flip in favor of QLoRA?
