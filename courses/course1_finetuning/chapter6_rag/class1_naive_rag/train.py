"""Course 1 / ch6 / class 1 — naive RAG: encoder retrieval + decoder generation.

No training step. Embed the corpus once, retrieve top-k per question via
cosine similarity, format a chat prompt with the retrieved passages, and
generate with SmolLM2-Instruct. Reports retrieval recall@k and answer
substring match.
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset

from shared.backbones import load_backbone
from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _to_id_list(raw) -> list[str]:
    """Normalize relevant_passage_ids to a list of strings.

    Datasets ship this column in many shapes:
      - list of ints/strings: ``[123, 456]`` or ``["123", "456"]``
      - stringified list literal: ``"[123, 456, ...]"`` (rag-mini-bioasq)
      - comma- or space-separated string: ``"123, 456"`` or ``"123 456"``
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return s.split()


def _build_prompt(tokenizer, question: str, passages: list[str]) -> str:
    """Render a chat prompt with retrieved context using the model's chat template."""
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    messages = [
        {"role": "system",
         "content": "You answer biomedical questions concisely using only the "
                    "provided passages. If the passages don't contain the answer, "
                    "say you don't know."},
        {"role": "user",
         "content": f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch6.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    # Auto-derive method tag from top_k.
    top_k = int(cfg["retrieval"]["top_k"])
    if cfg.get("method", "").startswith("rag-k"):
        cfg["method"] = f"rag-k{top_k}"

    # --- Load encoder + decoder ---------------------------------------------
    log.info("loading retriever: %s", cfg["backbone"])
    enc = load_backbone(cfg["backbone"])
    log.info("loading generator: %s", cfg["generator"])
    dec = load_backbone(cfg["generator"])
    if dec.kind != "decoder":
        raise ValueError(f"generator must be a decoder backbone, got {dec.kind}")

    # --- Load + cap corpus, then embed --------------------------------------
    ds_cfg = cfg["dataset"]
    corpus = load_dataset(
        ds_cfg["corpus_hf_id"], ds_cfg["corpus_config"], split=ds_cfg["corpus_split"]
    )
    cap_corpus = cfg["limits"][cfg["mode"]]["corpus_n"]
    if cap_corpus is not None:
        corpus = corpus.select(range(min(cap_corpus, len(corpus))))
    corpus_ids = [str(x) for x in corpus[ds_cfg["corpus_id_field"]]]
    corpus_texts = list(corpus[ds_cfg["corpus_text_field"]])
    log.info("corpus size: %d passages", len(corpus_texts))

    log.info("embedding corpus (this is the slow step)...")
    if enc.kind == "sentence-encoder":
        corpus_emb = enc.model.encode(
            corpus_texts, convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False, batch_size=64,
        )
    else:
        raise ValueError(
            f"retriever must be a sentence-encoder, got {enc.kind} for {cfg['backbone']}"
        )
    log.info("corpus embeddings: shape=%s dtype=%s", tuple(corpus_emb.shape), corpus_emb.dtype)

    # --- Load + cap QA ------------------------------------------------------
    qa = load_dataset(
        ds_cfg["qa_hf_id"], ds_cfg["qa_config"], split=ds_cfg["qa_split"]
    )
    qa = qa.shuffle(seed=cfg["seed"])
    cap_eval = cfg["limits"][cfg["mode"]]["eval_n"]
    if cap_eval is not None:
        qa = qa.select(range(min(cap_eval, len(qa))))
    log.info("eval questions: %d", len(qa))

    q_field = ds_cfg["qa_question_field"]
    a_field = ds_cfg["qa_answer_field"]
    rel_field = ds_cfg["qa_relevant_ids_field"]

    # --- Per-question retrieval + generation -------------------------------
    n_total = 0
    n_recall_hits = 0
    n_substring_hits = 0
    top_scores: list[float] = []
    examples: list[dict] = []

    id_to_index = {pid: i for i, pid in enumerate(corpus_ids)}

    for row in qa:
        question = row[q_field]
        gold_answer = (row[a_field] or "").strip()
        relevant_ids = _to_id_list(row[rel_field])

        # Retrieve.
        q_emb = enc.model.encode(
            [question], convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = (q_emb @ corpus_emb.T).squeeze(0)  # [N]
        top_vals, top_idx = torch.topk(sims, k=min(top_k, sims.numel()))
        top_idx_list = top_idx.cpu().tolist()
        top_passages = [corpus_texts[i] for i in top_idx_list]
        top_pids = [corpus_ids[i] for i in top_idx_list]
        top_scores.append(float(top_vals[0].item()))

        recall_hit = any(rid in top_pids for rid in relevant_ids) if relevant_ids else False
        n_recall_hits += int(recall_hit)

        # Generate.
        prompt = _build_prompt(dec.tokenizer, question, top_passages)
        inputs = dec.tokenizer(prompt, return_tensors="pt").to(dec.model.device)
        with torch.no_grad():
            out_ids = dec.model.generate(
                **inputs,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                do_sample=cfg["generation"]["do_sample"],
                temperature=cfg["generation"]["temperature"],
                pad_token_id=dec.tokenizer.eos_token_id,
            )
        new_text = dec.tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        substring_hit = bool(gold_answer) and gold_answer.lower() in new_text.lower()
        n_substring_hits += int(substring_hit)

        n_total += 1
        if len(examples) < 3:
            examples.append({
                "question": question,
                "gold_answer": gold_answer,
                "top_passage_ids": top_pids,
                "relevant_ids": relevant_ids,
                "recall_hit": recall_hit,
                "substring_hit": substring_hit,
                "generation_preview": new_text[:240],
            })

    recall_at_k = n_recall_hits / n_total if n_total else 0.0
    substring_match = n_substring_hits / n_total if n_total else 0.0
    mean_top_score = float(sum(top_scores) / len(top_scores)) if top_scores else 0.0
    log.info("retrieval_recall_at_%d=%.4f answer_substring_match=%.4f mean_top_score=%.4f",
             top_k, recall_at_k, substring_match, mean_top_score)
    for ex in examples:
        log.info("  example: hit_recall=%s hit_sub=%s q=%s gold=%s gen=%s",
                 ex["recall_hit"], ex["substring_hit"],
                 ex["question"][:80], ex["gold_answer"][:80],
                 ex["generation_preview"][:120].replace("\n", " "))

    metrics = {
        "retrieval_recall_at_k": float(recall_at_k),
        "answer_substring_match": float(substring_match),
        "n_questions_evaluated": int(n_total),
        "mean_top_passage_score": float(mean_top_score),
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "retriever": cfg["backbone"],
            "generator": cfg["generator"],
            "top_k": top_k,
            "corpus_n": len(corpus_texts),
            "examples": examples,
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
