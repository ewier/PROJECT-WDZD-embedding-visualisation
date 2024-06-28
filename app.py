import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP
from trimap import TRIMAP
from clastering import Clusterer, plot_embeddings, plot_labels
from embeddings_and_vector_search import ModelOptions, DatasetOptions, EmbeddingModel

dataset_names = {
    DatasetOptions.Symptoms: 1,
    DatasetOptions.Quotes: 2,
    DatasetOptions.Coments: 3
}

st.title('Wizualizacja osadzeń tekstowych')


def select_dataset():
    selected_dataset_name = st.selectbox("Wybierz dataset:", list(dataset_names.keys()))
    return dataset_names[selected_dataset_name]


def select_model():
    model_options = [value for attr, value in ModelOptions.__dict__.items() if not attr.startswith('__')]
    return st.selectbox("Wybierz model do generowania osadzeń:", model_options)


def get_dataset_embedding(model):
    model_mapping = {
        ModelOptions.MiniLM: 1,
        ModelOptions.BERT: 2,
    }
    return model_mapping.get(model, 1)


def load_embeddings_and_labels(selected_model, model_id, dataset_id):
    model = EmbeddingModel(model_name=selected_model, model_id=model_id)
    embeddings_path = model.properties["documents"][f"embedding_{model_id}_{dataset_id}_file_path"]
    labels_path = model.properties["documents"][f"labels{dataset_id}_file_path"]

    embeddings = pd.read_csv(embeddings_path, header=None)
    labels = pd.read_csv(labels_path, header=None, names=['label'])

    return embeddings, labels


def select_reduction_method():
    return st.selectbox("Wybierz metodę redukcji wymiarów:", ['UMAP', 't-SNE', 'PaCMAP', 'TriMAP'])


def configure_reducer(method, docs_length):
    if method == 't-SNE':
        perplexity = st.slider('Perplexity', min_value=5, max_value=docs_length - 1, value=5)
        learning_rate = st.slider('Learning Rate', min_value=10, max_value=1000, value=200)
        metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        init = st.selectbox('Initialization Method', ['random', 'pca'])
        return TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, metric=metric, init=init)

    if method == 'UMAP':
        n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=docs_length - 1, value=2)
        min_dist = st.slider('Minimum Distance', min_value=0.0, max_value=1.0, value=0.1)
        spread = st.slider('Spread', min_value=0.5, max_value=5.0, value=1.0)
        metric = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        return UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, metric=metric)

    if method == 'PaCMAP':
        n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=docs_length - 1, value=2)
        mn_ratio = st.slider('MN Ratio', min_value=0.1, max_value=1.0, value=0.5)
        fp_ratio = st.slider('FP Ratio', min_value=1.0, max_value=2.0, value=2.0)
        return PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=mn_ratio, FP_ratio=fp_ratio)

    if method == 'TriMAP':
        n_inliers = st.slider('Number of Inliers', min_value=5, max_value=docs_length - 1, value=5)
        n_outliers = st.slider('Number of Outliers', min_value=5, max_value=docs_length - 1, value=5)
        n_random = st.slider('Number of Random Points', min_value=1, max_value=docs_length - 1, value=1)
        distance = st.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine'])
        return TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random, distance=distance)


def generate_plot(n_clusters, plot_title, reducer, embeddings, labels):
    reduced_embeddings = reducer.fit_transform(embeddings.values)
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['label'] = labels['label']

    clusterer = Clusterer(n_clusters=n_clusters)
    df['cluster'] = clusterer.fit_predict(df)
    st.session_state['fig'] = plot_embeddings(df, plot_title)
    st.session_state['label_fig'] = plot_labels(df, plot_title)


def main():
    selected_dataset = select_dataset()
    selected_model = select_model()

    if st.button('Wybierz metodę osadzenia'):
        model_id = get_dataset_embedding(selected_model)
        embeddings, labels = load_embeddings_and_labels(selected_model, model_id, selected_dataset)

        st.session_state['docs'] = embeddings
        st.session_state['labels'] = labels

    if 'docs' in st.session_state and 'labels' in st.session_state:
        method = select_reduction_method()
        reducer = configure_reducer(method, len(st.session_state['docs']))

        number_of_clusters = st.slider('Wybierz liczbę klastrów:', min_value=2, max_value=100, value=5)
        plot_title = st.text_input('Wprowadź tytuł wykresu', '{} {} {}'.format(method, "Projection of",
                                                                               list(dataset_names.keys())[
                                                                                   list(dataset_names.values()).index(
                                                                                       selected_dataset)]))

        if st.button('Generuj wykres'):
            generate_plot(number_of_clusters, plot_title, reducer, st.session_state['docs'], st.session_state['labels'])

            if 'fig' in st.session_state:
                st.plotly_chart(st.session_state['fig'])
                st.plotly_chart(st.session_state['label_fig'])


if __name__ == "__main__":
    main()
