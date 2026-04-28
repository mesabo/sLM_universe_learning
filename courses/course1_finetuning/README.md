# Course 1 — Fine-tuning small language models

The longest course. Six chapters, each one a different "knob" you can turn when adapting a small model to a task.

| Chapter | Topic | Why |
|---|---|---|
| `chapter1_full_ft/` | Full fine-tuning (encoder + decoder) | The baseline everything else is compared against |
| `chapter2_lora/` | LoRA (PEFT) | The first thing you'll actually use in practice |
| `chapter3_qlora/` | QLoRA + 4-bit (PEFT + bitsandbytes) | Train SmolLM2-360M on a 12 GB card |
| `chapter4_dpo/` | Preference tuning (DPO / ORPO / KTO) | When labels become "A > B" instead of class indices |
| `chapter5_embedding_ft/` *(v2)* | Sentence-encoder contrastive fine-tuning | Bring MiniLM/BGE/GTE up to your retrieval distribution |
| `chapter6_rag/` *(v2)* | Retrieval-augmented generation | When NOT to fine-tune at all — and when to fine-tune the retriever |
| `chapter7_eval_discipline/` *(v2)* | Held-out, contamination, per-(method, backbone) JSON matrix | Make results trustworthy |

Every chapter declares a metric band; results land at `results/full/<backbone>/course1_finetuning/<class>/<task>/<method>.json`.
