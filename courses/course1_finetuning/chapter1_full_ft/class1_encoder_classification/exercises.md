# Exercises — Course 1 · ch1 · class 1

## 1. Warm-up — sweep the backbone

Run smoke mode against all three encoder backbones and tabulate accuracy:

```bash
for bb in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 thenlper/gte-small; do
  python courses/.../train.py --config courses/.../configs/default.yaml backbone="$bb"
done
```

Then `from shared.eval_harness import aggregate; aggregate()`.

## 2. Apply — linear probe baseline

Add a `freeze_base: true` knob to the YAML; in `train.py`, call `shared.training.freeze_base(model.base_model)` before training. How much accuracy do you lose vs full FT? How much memory do you save?

## 3. Stretch — different head architecture

Replace the default `nn.Linear(hidden, num_labels)` head with `nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_labels))`. Fork `model.classifier`. Does the deeper head help on AG News? Why or why not? (Hint: AG News is *easy* and the encoder is doing most of the work.)
