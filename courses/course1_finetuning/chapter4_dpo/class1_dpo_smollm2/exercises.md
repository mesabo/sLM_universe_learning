# Exercises — Course 1 · ch 4 · class 1 (DPO)

## 1. Warm-up — sweep beta

```bash
for b in 0.01 0.1 0.5; do
  bash run.sh --config configs/default.yaml dpo.beta=$b
done
```

Tabulate `eval_loss`, `rewards_margin`, `rewards_accuracy` per `beta`. Lower beta lets the policy drift more — does that always raise `rewards_margin`? At what beta does training collapse?

## 2. Apply — generate before & after

Take a prompt from the held-out set. Generate with the as-loaded SmolLM2-Instruct (the reference) and with the saved policy from this class. Are the policy's generations recognizably more "preferred-style"? If you can't tell the difference, smoke training is too short.

## 3. Stretch — DPO on a LoRA adapter

Wrap the policy with `peft.get_peft_model(...)` before passing to `DPOTrainer` (no `ref_model` needed — TRL uses the base as the reference). Re-run smoke. How does memory cost change? Eval reward margin?
