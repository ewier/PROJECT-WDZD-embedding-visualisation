from elasticsearch import Elasticsearch
import json
import time
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer


class PropertiesLoader:
    @staticmethod
    def read_properties(path="properties/properties.json"):
        f = open(path)
        data = json.load(f)
        return data


class ModelOptions:
    # Here are all the supproted models
    MiniLM = "miniLM"
    BERT = "bert"



class EmbeddingModel:

    def __init__(self, model_name):
        self.model = model_name
        self.properties = PropertiesLoader.read_properties()

    def get_embedding(self):
        document_text = []
        with open(self.properties["documents"]["document_file_path"], "r") as document_file:
            document_text = document_file.readlines()
        initial_time = time.time()
        if self.model == ModelOptions.MiniLM:
            embedding = self._get_miniLM(document_text)
        elif self.model == ModelOptions.BERT:
            embedding = self._get_bert(document_text)
        else:
            raise Exception("This model name is not supported.")
        total_time = time.time() - initial_time
        print(f'Documents embedding finished in {total_time} second(s)\n')
        with open(self.properties["documents"]["embedding_file_path"], "w") as embedding_file:
            for emb in embedding:
                embedding_file.write(','.join([str(i) for i in emb]))
                embedding_file.write('\n')

    @staticmethod
    def _get_miniLM(texts: list[str]):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = []
        for text in texts:
            embeddings.append(model.encode(text))
        return embeddings
    @staticmethod
    def _get_bert(texts: list[str]):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = []
        for text in texts:
            embeddings.append(model.encode(text))
        return embeddings



class TextIndexer:
    def __init__(self) -> None:
        self.properties = PropertiesLoader.read_properties()
        user, password = self.properties['user'], self.properties['password']
        self.client = Elasticsearch('http://localhost:9200', basic_auth=(user, password))
        self.BATCH_SIZE = self.properties['batch_size']
        self.INDEX_NAME = self.properties['index_name']

    def _create_index(self):
        self.client.options(ignore_status=[400, 404]).indices.delete(index=self.INDEX_NAME)
        self.client.indices.create(index=self.INDEX_NAME)

    def _index_documents(self):
        document_source = self.properties['documents']['document_file_path']
        embedding_source = self.properties['documents']['embedding_file_path']
        with open(document_source, "r") as documents_file:
            with open(embedding_source, "r") as vectors_file:
                documents = []
                for index, (document, vector_string) in enumerate(zip(documents_file, vectors_file)):
                    vector = [float(w) for w in vector_string.split(",")]

                    doc = {
                        "_id": str(index),
                        "general_text": document,
                        "general_text_vector": vector
                    }
                    documents.append(doc)

                    if index % self.BATCH_SIZE == 0 and index != 0:
                        indexing = bulk(self.client, documents, index=self.INDEX_NAME)
                        documents = []
                        print(f"Indexing failed for {len(indexing[1])} items")
                if documents:
                    bulk(self.client, documents, index=self.INDEX_NAME)
                print("Finished")
    
    def create_and_index(self):
        self._create_index()
        initial_time = time.time()
        self._index_documents()
        total_time = time.time() - initial_time
        print(f'Document indexing finished in {total_time} second(s)\n')
        
        

if __name__ == "__main__":
    # Embedding documents
    model = EmbeddingModel(model_name=ModelOptions.MiniLM)
    model.get_embedding()

    # Indexing documents
    indexer = TextIndexer()
    indexer.create_and_index()
