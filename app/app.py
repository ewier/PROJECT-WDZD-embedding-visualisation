import os
import sys
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP
from trimap import TRIMAP
from visualization import plot_embeddings
from embeddings_and_vector_search import ModelOptions, EmbeddingModel

st.title('Wizualizacja osadzeń tekstowych')

model_options = [value for attr, value in ModelOptions.__dict__.items() if not attr.startswith('__')]
selected_model = st.selectbox("Wybierz model do generowania osadzeń:", model_options)

model = EmbeddingModel(model_name=selected_model)
model.get_embedding()

embeddings = pd.read_csv(model.properties["documents"]["embedding_file_path"], header=None)

st.session_state['docs'] = embeddings
method = st.selectbox("Wybierz metodę redukcji wymiarów:", ['UMAP', 't-SNE', 'PaCMAP', 'TriMAP'])

if method == 't-SNE':
    perplexity = st.slider('Perplexity', min_value=5, max_value=len(embeddings)-1, value=5)
    learning_rate = st.slider('Learning Rate', min_value=10, max_value=1000, value=200)
    metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
    init = st.selectbox('Initialization Method', ['random', 'pca'])
elif method == 'UMAP':
    n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=len(embeddings)-1, value=2)
    min_dist = st.slider('Minimum Distance', min_value=0.0, max_value=1.0, value=0.1)
    spread = st.slider('Spread', min_value=1.0, max_value=5.0, value=1.0)
    metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
elif method == 'PaCMAP':
    n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=len(embeddings)-1, value=2)
    MN_ratio = st.slider('MN Ratio', min_value=0.1, max_value=1.0, value=0.5)
    FP_ratio = st.slider('FP Ratio', min_value=1.0, max_value=2.0, value=2.0)
    init = st.selectbox('Initialization Method', ['random', 'pca'])
elif method == 'TriMAP':
    n_inliers = st.slider('Number of Inliers', min_value=5, max_value=len(embeddings)-1, value=5)
    n_outliers = st.slider('Number of Outliers', min_value=5, max_value=len(embeddings)-1, value=5)
    n_random = st.slider('Number of Random Points', min_value=1, max_value=len(embeddings)-1, value=1)
    distance = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])

if st.button('Generuj wizualizację'):
    embeddings = st.session_state.get('docs', pd.DataFrame())
    if method == 't-SNE':
        # perplexity_value = max(len(embeddings) / 3, 5)
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, metric=metric, init=init)
    elif method == 'UMAP':
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, metric=metric)
    elif method == 'PaCMAP':
        reducer = PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio, init=init)
    elif method == 'TriMAP':
        reducer = TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random, distance=distance)
    else:
        st.write("Metoda nie jest jeszcze zaimplementowana")

    reduced_embeddings = reducer.fit_transform(embeddings.values)
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])

    fig = plot_embeddings(df)
    st.pyplot(fig)

    st.session_state['fig'] = fig

if 'fig' in st.session_state:
    if st.button('Wybierz folder do zapisu obrazu'):
        os.system("python select_folder.py")
        with open("selected_folder.json", "r") as f:
            folder_data = json.load(f)
        st.session_state['folder_path'] = folder_data['folder_path']
        st.text_input('Wybrana ścieżka:', folder_data['folder_path'])

    if 'folder_path' in st.session_state and st.session_state['folder_path']:
        st.session_state['fig'].savefig(f"{st.session_state['folder_path']}/visualization.png")
        st.success(f"Obraz został zapisany w {st.session_state['folder_path']}/visualization.png")


