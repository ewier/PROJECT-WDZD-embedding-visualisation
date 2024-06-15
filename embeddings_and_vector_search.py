from elasticsearch import Elasticsearch
import json
import time
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


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

    def get_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        full_dataset = dataset['train']
        document_text = [ex['input_text'] for ex in full_dataset]
        document_label = [ex['output_text'] for ex in full_dataset]
        return document_text, document_label

    def get_embedding(self, dataset_name):
        document_text, document_labels = self.get_dataset(dataset_name)
        initial_time = time.time()

        match self.model:
            case ModelOptions.MiniLM:
                embedding = self._get_miniLM(document_text)
            case ModelOptions.BERT:
                embedding = self._get_bert(document_text)

        total_time = time.time() - initial_time
        print(f'Documents embedding finished in {total_time} second(s)\n')
        with open(self.properties["documents"]["document_file_path"], "w") as documents_file:
            documents_file.write('\n'.join([str(i) for i in document_text]))
        with open(self.properties["documents"]["embedding_file_path"], "w") as embedding_file:
            for emb in embedding:
                embedding_file.write(','.join([str(i) for i in emb]))
                embedding_file.write('\n')
        with open(self.properties["documents"]["labels_file_path"], "w") as label_file:
            label_file.write('\n'.join([str(i) for i in document_labels]))

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

    def _get_documents(self):
        document_source = self.properties['documents']['document_file_path']
        embedding_source = self.properties['documents']['embedding_file_path']
        labels_source = self.properties['documents']['labels_file_path']
        with open(document_source, "r") as documents_file:
            documnets_text = documents_file.readlines()
        with open(embedding_source, "r") as embedding_file:
            raw_embeddings = embedding_file.readlines()
            embeddings = [[float(value) for value in line.split(',')] for line in raw_embeddings]
        with open(labels_source, "r") as labels_file:
            labels = labels_file.readlines()
        return documnets_text, embeddings, labels

    def _index_documents(self):
        documnets_text, embeddings, labels = self._get_documents()
        documents = []
        for index, (document, vector, label) in enumerate(zip(documnets_text, embeddings, labels)):
            doc = {
                "_id": str(index),
                "general_text": document,
                "general_text_vector": vector,
                "label": label
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
