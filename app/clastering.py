from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


class Clusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit_predict(self, embeddings):
        return self.model.fit_predict(embeddings[['x', 'y']])


def plot_embeddings(df):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['label'])
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    return fig