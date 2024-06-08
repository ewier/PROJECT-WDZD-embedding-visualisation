import os
import sys
import tempfile
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from embeddings_and_vector_search import EmbeddingModel

def generate_embeddings(file, model_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file.flush()
        tmp_file_path = tmp_file.name

    model = EmbeddingModel(model_name=model_name)
    model.properties['documents']['document_file_path'] = tmp_file_path
    model.get_embedding()

    embeddings = pd.read_csv(model.properties["documents"]["embedding_file_path"], header=None)
    os.remove(tmp_file_path)
    return embeddings
