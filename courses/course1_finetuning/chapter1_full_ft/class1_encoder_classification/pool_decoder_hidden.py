# pool_decoder_hidden.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
inputs = tok("I love ML", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, return_dict=True)
last = out.hidden_states[-1]            # shape (batch, seq_len, hidden)
pooled = last.mean(dim=1).squeeze(0)    # shape (hidden,)
print("last hidden shape:", last.shape)
print("pooled shape:", pooled.shape)