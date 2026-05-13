import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use local snapshot to avoid remote downloads
LOCAL_SMOL = os.path.join(os.path.dirname(__file__), "../../../../.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_SMOL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(LOCAL_SMOL)
model.eval()

text = "The quick brown fox."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, return_dict=True)
# hidden_states[-1]: (batch, seq_len, hidden_size)
h = out.hidden_states[-1]

# 1) simple mean pooling over tokens
emb_mean = h.mean(dim=1)            # shape: (batch, hidden_size)

# 2) last non-padding token pooling (often used for decoder-as-encoder)
input_ids = inputs["input_ids"]
last_idx = (input_ids != tokenizer.pad_token_id).sum(dim=1) - 1
emb_last = h[torch.arange(h.size(0)), last_idx, :]  # shape: (batch, hidden_size)

print(emb_mean.shape, emb_last.shape)