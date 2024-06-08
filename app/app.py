import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
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

if st.button('Generuj wizualizację'):
    embeddings = st.session_state.get('docs', pd.DataFrame())
    if method == 't-SNE':
        perplexity_value = max(len(embeddings) / 3, 5)
        reducer = TSNE(n_components=2, perplexity=perplexity_value)
    elif method == 'UMAP':
        reducer = UMAP(n_components=2)
    else:
        st.write("Metoda nie jest jeszcze zaimplementowana")

    reduced_embeddings = reducer.fit_transform(embeddings.values)
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    st.pyplot(plot_embeddings(df))
