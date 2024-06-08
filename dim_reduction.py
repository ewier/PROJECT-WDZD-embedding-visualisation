import umap
import pacmap
import trimap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

embeddings = np.loadtxt('docs/embeddings.txt', delimiter=',')

# UMAP
umap_reducer = umap.UMAP(n_neighbors=min(15, len(embeddings) - 1))
umap_embeddings = umap_reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=1)
plt.title('UMAP')
plt.show()

# PaCMAP
pacmap_reducer = pacmap.PaCMAP()
pacmap_embeddings = pacmap_reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(pacmap_embeddings[:, 0], pacmap_embeddings[:, 1], s=1)
plt.title('PaCMAP')
plt.show()

# TriMAP
trimap_reducer = trimap.TRIMAP(n_inliers=min(10, len(embeddings) - 2))
trimap_embeddings = trimap_reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(trimap_embeddings[:, 0], trimap_embeddings[:, 1], s=1)
plt.title('TriMAP')
plt.show()

# t-SNE
tsne_reducer = TSNE(n_components=2, perplexity=5)
tsne_embeddings = tsne_reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=1)
plt.title('t-SNE')
plt.show()
