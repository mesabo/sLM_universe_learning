# encode_with_st.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # local snapshot ok
s = "I love ML"
emb = model.encode(s, convert_to_tensor=False)
print("shape:", emb.shape)
# use cosine similarity between embeddings for comparison