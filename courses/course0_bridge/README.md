# Course 0 — Bridge

Just-enough Hugging Face + transformer refresher to start fine-tuning small LMs in Course 1. Deliberately tight: we do **not** re-derive attention or build a transformer from scratch. We learn the APIs and the encoder-vs-decoder distinction, then move on.

| Chapter | Topic | Classes |
|---|---|---|
| `chapter1_hf_tour/` | The HF ecosystem (`AutoModel`, `AutoTokenizer`, `pipeline`, `Trainer`) | `class1_automodel` |
| `chapter2_encoder_vs_decoder/` | Loading and using both kinds side by side | `class1_side_by_side` |
| `chapter3_tokenization/` *(coming in v1)* | BPE, WordPiece, chat templates | TBD |

References used in this course (official only):
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [sentence-transformers documentation](https://www.sbert.net/)
