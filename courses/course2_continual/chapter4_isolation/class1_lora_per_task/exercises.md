# Exercises — Course 2 · ch 4 · class 1 (LoRA-per-task isolation)

## 1. Warm-up — verify the head IS per-adapter

In a Python shell after `bash run.sh`:

```python
from peft import PeftModel
# ... load the saved model with both adapters ...
m.set_adapter("ag_news"); print(m.classifier.modules_to_save["ag_news"].weight[0, :5])
m.set_adapter("emotion"); print(m.classifier.modules_to_save["emotion"].weight[0, :5])
```

Are the two heads' first row of weights different? They should be — that's `modules_to_save=["classifier"]` doing its job.

## 2. Apply — drop `modules_to_save`

Edit the YAML to set `lora.modules_to_save: []`. Re-run smoke. What happens to BWT? (Hint: the head becomes shared again, and Task B's training rewrites it. Forgetting returns even though the LoRA matrices are isolated.)

## 3. Stretch — three tasks

Add a third task to `tasks:` (e.g. `SetFit/sst2` mapped to 2-of-4 labels). Modify the train loop to handle N tasks: at stage k, add `adapter_k` and train. At final eval, swap per task. Does BWT stay near 0 with 3 tasks?
