import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from visualization import plot_embeddings
from data_loader import generate_embeddings
from embeddings_and_vector_search import ModelOptions

st.title('Wizualizacja osadzeń tekstowych')

uploaded_file = st.file_uploader("Wybierz plik tekstowy do osadzenia", type=['txt'])
model_options = [value for attr, value in ModelOptions.__dict__.items() if not attr.startswith('__')]
selected_model = st.selectbox("Wybierz model do generowania osadzeń:", model_options)

if uploaded_file is not None and st.button('Generuj osadzenia'):
    embeddings = generate_embeddings(uploaded_file, selected_model)
    st.session_state['docs'] = embeddings

method = st.selectbox("Wybierz metodę redukcji wymiarów:", ['UMAP', 't-SNE', 'PaCMAP', 'TriMAP'])

if st.button('Generuj wizualizację'):
    embeddings = st.session_state.get('docs', pd.DataFrame())
    if not embeddings.empty:
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
    else:
        st.write("Proszę najpierw wygenerować osadzenia.")
