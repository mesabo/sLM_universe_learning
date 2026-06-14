"""Unified LangChain LLM client for Course 4.

Returns a ``BaseChatModel`` backed by either a local HuggingFace sLM or the
OpenAI API, depending on ``cfg["provider"]`` and whether ``OPENAI_API_KEY`` is
present in the environment.

Usage::

    from shared.llm_client import get_llm
    llm = get_llm(cfg)          # returns BaseChatModel
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from typing import Any


def get_llm(cfg: dict[str, Any]):  # noqa: ANN201
    """Return a LangChain BaseChatModel based on ``cfg``.

    Priority:
    1. ``cfg["provider"] == "openai"`` AND ``OPENAI_API_KEY`` set → ChatOpenAI
    2. ``cfg["provider"] == "huggingface"`` (or any other / missing) → local HF pipeline
    """
    provider = cfg.get("provider", "huggingface")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if provider == "openai" and api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.get("openai_model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_new_tokens", 256),
            api_key=api_key,
        )

    # Default: local HF backbone
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    backbone = cfg.get("backbone", "HuggingFaceTB/SmolLM2-135M-Instruct")
    hf_home = os.environ.get("HF_HOME", ".cache/huggingface")

    tokenizer = AutoTokenizer.from_pretrained(backbone, cache_dir=hf_home)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(backbone, cache_dir=hf_home)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.get("max_new_tokens", 128),
        temperature=cfg.get("temperature", 0.1),
        do_sample=cfg.get("temperature", 0.1) > 0,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_embedding_model(cfg: dict[str, Any]):  # noqa: ANN201
    """Return a LangChain Embeddings object.

    Uses ``langchain_huggingface.HuggingFaceEmbeddings`` with the sentence-encoder
    backbone (always local, no API needed).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embed_backbone = cfg.get(
        "embed_backbone", "sentence-transformers/all-MiniLM-L6-v2"
    )
    hf_home = os.environ.get("HF_HOME", ".cache/huggingface")

    return HuggingFaceEmbeddings(
        model_name=embed_backbone,
        cache_folder=hf_home,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
