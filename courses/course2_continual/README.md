# Course 2 — Preventing catastrophic forgetting

Catastrophic forgetting is the phenomenon where a neural net, after being fine-tuned on a new task B, loses most of what it learned on a prior task A. Course 1 showed how to fine-tune; Course 2 shows what fine-tuning *destroys* and what to do about it.

| Chapter | Topic | Why |
|---|---|---|
| `chapter1_measurement/` | Define the phenomenon, measure it (BWT / FWT / avg-acc) | Without numbers, you can't fix it |
| `chapter2_replay/` *(v3)* | Experience replay, mixing ratios | The cheapest defense, often the strongest |
| `chapter3_regularization/` *(v3)* | EWC / MAS / L2-SP — penalize drift in important params | The historical workhorse; teaches Fisher information |
| `chapter4_isolation/` *(v3)* | Parameter isolation: LoRA per task, multi-adapter loading | The modern PEFT-era answer |
| `chapter5_recipes/` *(v3)* | Combine ch2-4, ablation matrix across all 5 backbones | Show what actually wins |

Every chapter uses the metrics defined in `shared/continual.py` (BWT / FWT / average accuracy from Lopez-Paz & Ranzato, NeurIPS 2017). Results land at `results/full/<backbone>/course2_continual/<class>/<task>/<method>.json`.
