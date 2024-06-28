from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px


class Clusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit_predict(self, embeddings):
        return self.model.fit_predict(embeddings[['x', 'y']])


def plot_embeddings(df, title='Plot'):
    fig = px.scatter(
        df, x='x', y='y', color='cluster',
        hover_data=['label'],  #: True, 'x': False, 'y': False, 'cluster': False},
        title=title,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    return fig


def plot_labels(df, title='Plot'):
    fig = px.scatter(
        df, x='x', y='y',
        hover_data=['label'],  #: True, 'x': False, 'y': False, 'cluster': False},
        title=title,
        color=df['label']
    )
    return fig