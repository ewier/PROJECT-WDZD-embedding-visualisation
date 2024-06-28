import os
import sys
import json
import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP
from trimap import TRIMAP
from clastering import Clusterer, plot_embeddings, plot_labels
import tkinter as tk
from tkinter import filedialog

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from embeddings_and_vector_search import ModelOptions, EmbeddingModel

dataset_names = {
    'gretelai/symptom_to_diagnosis': 1,
    'Abirate/english_quotes': 2,
    'Yelp/yelp_review_full': 3,
    'Coments': 4
}

st.title('Wizualizacja osadzeń tekstowych')

selected_dataset_name = st.selectbox("Wybierz dataset:", list(dataset_names.keys()))
selected_dataset = dataset_names[selected_dataset_name]

model_options = [value for attr, value in ModelOptions.__dict__.items() if not attr.startswith('__')]
selected_model = st.selectbox("Wybierz model do generowania osadzeń:", model_options)

def get_dataset_embedding(model):
    match model:
        case ModelOptions.MiniLM:
            model_id = 1
        case ModelOptions.BERT:
            model_id = 2
    return model_id

if st.button('Wybierz metodę osadzenia'):
    model_id = get_dataset_embedding(selected_model)
    model = EmbeddingModel(model_name=selected_model, model_id=model_id)
    embeddings = pd.read_csv(model.properties["documents"][f"embedding_{model_id}_{selected_dataset}_file_path"], header=None)
    labels = pd.read_csv(model.properties["documents"][f"labels{selected_dataset}_file_path"], header=None, names=['label'])

    st.session_state['docs'] = embeddings
    st.session_state['labels'] = labels

if 'docs' in st.session_state and 'labels' in st.session_state:
    method = st.selectbox("Wybierz metodę redukcji wymiarów:", ['UMAP', 't-SNE', 'PaCMAP', 'TriMAP'])

    if method == 't-SNE':
        perplexity = st.slider('Perplexity', min_value=5, max_value=len(st.session_state['docs']) - 1, value=5)
        learning_rate = st.slider('Learning Rate', min_value=10, max_value=1000, value=200)
        metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        init = st.selectbox('Initialization Method', ['random', 'pca'])
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, metric=metric, init=init)
    elif method == 'UMAP':
        n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=len(st.session_state['docs']) - 1, value=2)
        min_dist = st.slider('Minimum Distance', min_value=0.0, max_value=1.0, value=0.1)
        spread = st.slider('Spread', min_value=1.0, max_value=5.0, value=1.0)
        metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, metric=metric)
    elif method == 'PaCMAP':
        n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=len(st.session_state['docs']) - 1, value=2)
        MN_ratio = st.slider('MN Ratio', min_value=0.1, max_value=1.0, value=0.5)
        FP_ratio = st.slider('FP Ratio', min_value=1.0, max_value=2.0, value=2.0)
        reducer = PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
    elif method == 'TriMAP':
        n_inliers = st.slider('Number of Inliers', min_value=5, max_value=len(st.session_state['docs']) - 1, value=5)
        n_outliers = st.slider('Number of Outliers', min_value=5, max_value=len(st.session_state['docs']) - 1, value=5)
        n_random = st.slider('Number of Random Points', min_value=1, max_value=len(st.session_state['docs']) - 1, value=1)
        distance = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        reducer = TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random, distance=distance)

    number_of_clusters = st.slider('Wybierz liczbę klastrów:', min_value=2, max_value=10, value=5)
    plot_title = st.text_input('Wprowadź tytuł wykresu', selected_dataset_name)

    def generate_plot(n_clusters, plot_title):
        embeddings = st.session_state.get('docs', pd.DataFrame())
        labels = st.session_state.get('labels', pd.DataFrame())
        if not labels.empty:
            labels.columns = ['label']

        reduced_embeddings = reducer.fit_transform(embeddings.values)
        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        df['label'] = labels['label']

        clusterer = Clusterer(n_clusters=n_clusters)
        df['cluster'] = clusterer.fit_predict(df)
        fig = plot_embeddings(df, plot_title)
        st.session_state['fig'] = fig
        st.session_state['label_fig'] = plot_labels(df, plot_title)

    if st.button('Generuj wykres'):
        generate_plot(number_of_clusters, plot_title)

        if st.session_state['fig']:
            st.plotly_chart(st.session_state['fig'])
            st.plotly_chart(st.session_state['label_fig'])

