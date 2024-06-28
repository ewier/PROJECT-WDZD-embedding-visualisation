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
    # Here are all the supported models
    MiniLM = "miniLM"
    BERT = "bert"

class DatasetOptions:
    # Here are all the supported datasets
    Symptoms = 'gretelai/symptom_to_diagnosis'
    Quotes = 'Abirate/english_quotes'
    Reviews = 'codyburker/yelp_review_sampled'


class EmbeddingModel:

    def __init__(self, model_name, model_id):
        self.model = model_name
        self.model_id = model_id
        self.properties = PropertiesLoader.read_properties()

    def get_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        match dataset_name:
            case DatasetOptions.Symptoms:
                full_dataset = dataset['train']
                document_text = [ex['input_text'] for ex in full_dataset]
                document_label = [ex['output_text'] for ex in full_dataset]
            case DatasetOptions.Reviews:
                full_dataset = dataset['train']
                document_text = [ex['text'] for ex in full_dataset]
                document_label = [ex['stars'] for ex in full_dataset]
            case DatasetOptions.Quotes:
                full_dataset = dataset['train']
                document_text = [ex['quote'] for ex in full_dataset]
                document_label = [ex['author'] for ex in full_dataset]            
        return document_text, document_label

    def get_embedding(self, dataset_name, dataset_id):
        document_text, document_labels = self.get_dataset(dataset_name)
        initial_time = time.time()

        match self.model:
            case ModelOptions.MiniLM:
                embedding = self._get_miniLM(document_text)
            case ModelOptions.BERT:
                embedding = self._get_bert(document_text)

        total_time = time.time() - initial_time
        print(f'Documents embedding finished in {total_time} second(s)\n')
        with open(self.properties["documents"][f"document{dataset_id}_file_path"], "w", encoding="utf-8") as documents_file:
            documents_file.write('\n'.join([str(i) for i in document_text]))
        with open(self.properties["documents"][f"embedding_{model_id}_{dataset_id}_file_path"], "w", encoding="utf-8") as embedding_file:
            for emb in embedding:
                embedding_file.write(','.join([str(i) for i in emb]))
                embedding_file.write('\n')
        with open(self.properties["documents"][f"labels{dataset_id}_file_path"], "w", encoding="utf-8") as label_file:
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

    def _get_documents(self, sources):
        document_source = sources['document_source']
        embedding_source = sources['embedding_source']
        labels_source = sources['labels_source']
        with open(document_source, "r", encoding="utf-8") as documents_file:
            documnets_text = documents_file.readlines()
        with open(embedding_source, "r", encoding="utf-8") as embedding_file:
            raw_embeddings = embedding_file.readlines()
            embeddings = [[float(value) for value in line.split(',')] for line in raw_embeddings]
        with open(labels_source, "r", encoding="utf-8") as labels_file:
            labels = labels_file.readlines()
        return documnets_text, embeddings, labels

    def _index_documents(self, sources):
        documnets_text, embeddings, labels = self._get_documents(sources)
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
    
    def create_and_index(self, model_id, dataset_id):
        sources = {
            "document_source": self.properties["documents"][f"document{dataset_id}_file_path"],
            "embedding_source": self.properties["documents"][f"embedding_{model_id}_{dataset_id}_file_path"],
            "labels_source": self.properties["documents"][f"labels{dataset_id}_file_path"],
        }
        self._create_index()
        initial_time = time.time()
        self._index_documents(sources)
        total_time = time.time() - initial_time
        print(f'Document indexing finished in {total_time} second(s)\n')
        
        

if __name__ == "__main__":
    # Embedding documents
    datasets = [DatasetOptions.Symptoms, DatasetOptions.Quotes, DatasetOptions.Reviews]
    models = [ModelOptions.BERT, ModelOptions.MiniLM]
    # finish MiniLM
    dataset_id, dataset_name = 4, DatasetOptions.Reviews
    model_id, model_name = 2, ModelOptions.BERT
    print(f"MODEL {model_id}, DATASET {dataset_id}")
    model = EmbeddingModel(model_name=model_name, model_id=model_id)
    model.get_embedding(dataset_name, dataset_id)
    # Indexing documents
    indexer = TextIndexer()
    indexer.create_and_index(model_id, dataset_id)
