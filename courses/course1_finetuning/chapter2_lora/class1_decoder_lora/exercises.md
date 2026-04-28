# Exercises — Course 1 · ch2 · class 1

## 1. Warm-up — read the trainable-params print-out

Before training begins, PEFT prints something like `trainable params: 405,504 || all params: 134,920,448 || trainable%: 0.30`. Verify the math: with `r=16`, `target_modules=4`, hidden_size=576 — does 0.30% match? (Hint: 4 × 2 × r × hidden ≈ 73 728 per layer × 30 layers ≈ 2.2 M; the exact number depends on layer geometry.)

## 2. Apply — sweep `r`

Run the smoke with `lora.r=4`, `r=16`, `r=64`. Tabulate eval loss and final adapter file size (`du -sh checkpoints/.../adapter_model.safetensors`). At what rank does the adapter stop being "tiny"?

## 3. Stretch — multi-adapter loading

After two LoRA runs (e.g. one on smoltalk, one on a tiny subset of `databricks/databricks-dolly-15k`), load both onto one base via `peft.PeftModel.from_pretrained(base, adapter_a); model.load_adapter(adapter_b, "dolly")` and `model.set_adapter("dolly")`. Generate from each. This is the foundation for Course 2 ch4 (parameter isolation).
