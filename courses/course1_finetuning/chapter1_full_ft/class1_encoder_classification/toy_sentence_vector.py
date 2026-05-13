# toy_sentence_vector.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
E = np.array([[0.5,1.0,0.0],[1.1,-0.2,0.3],[0.0,0.6,-0.1]])
v = E.mean(axis=0)
print("token embeddings:\n", E)
print("sentence vector v:", v)
# project to 2D for visualization
pca = PCA(n_components=2)
pts = pca.fit_transform(np.vstack([E, v]))
tokens = ["I","love","ML","SENTENCE"]
plt.scatter(pts[:,0], pts[:,1])
for i,t in enumerate(tokens):
    plt.annotate(t, (pts[i,0], pts[i,1]))
plt.title("Tokens and pooled sentence vector (PCA 2D)")
plt.show()