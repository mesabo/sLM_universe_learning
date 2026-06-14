# Exercises — Class 1: Vector Stores

## Warm-up: Save and reload a FAISS index

After building a FAISS index, serialize it to disk with `faiss_store.save_local("index_dir")` and reload it with `FAISS.load_local("index_dir", embeddings, allow_dangerous_deserialization=True)`. Run a query on the reloaded index and verify the results are identical to the original. Measure the save and load time.

## Apply: Metadata filtering in ChromaDB

Add metadata fields `{"category": "retrieval"}` or `{"category": "training"}` to each document. Use Chroma's `where` filter to retrieve only documents with `category == "retrieval"`. Measure recall@3 for the filtered subset vs the full collection and explain any difference.

## Stretch: Recall vs k curve

For each `k` in `[1, 3, 5, 10]`, measure recall@k on the FAISS store using the labeled query set. Plot (or print) the curve. At what `k` does recall plateau? Explain this in terms of the corpus size and embedding quality.
