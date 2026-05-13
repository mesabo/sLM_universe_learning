from sentence_transformers import SentenceTransformer
import os

# Use the locally cached snapshot (avoid remote download)
LOCAL_MINILM = os.path.join(os.path.dirname(__file__), "../../../../.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

# SentenceTransformer accepts a local path
model = SentenceTransformer(LOCAL_MINILM)

text = "The quick brown fox."
# returns a ready-to-use embedding (1D tensor)
emb = model.encode(text, convert_to_tensor=True)    # shape: (hidden_size,)
print(emb.shape)