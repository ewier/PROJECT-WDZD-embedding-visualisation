# About the project

USING EMBEDDING TEXTS (E.G. CREATED AT 
HUGGINGFACEEMBEDDINGS HELP) AND ANY DATABASE 
VECTOR (NP. ELASTICSEARCH) CREATE IN STREAMLIT
 A DASHBOARD THAT WILL VISUALIZE THESE SETTLEMENTS 
UMAP, T-SNE, PACMAP, TRIMAP.

The goal of this project is to build an app that visualises text datasets using embedding models, a vector database and tools such as UMAP, T-SNE, PACMAP, or TRIMAP. The app was created in streamlit and allows the user to choose a dataset, a model and a dimentionality reduction technique.

# How to run

To run embedding and visualisation:
1. Install elasticsearch
2. Modify properties.json - add your elasticsearch password and update the document paths to match your environment
5. Initialize elasticsearch
6. Run embedding procedure and then indexing procedure


To run app: 
```
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

lub z Dockerfile:
```
docker build -t my-streamlit-app .
docker run -p 8502:8502 my-streamlit-app
```

# Datasets

Dataset sources:

## 1. Symptoms to diagnosis
The symptom to diagnosis dataset is avaliable [here](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)

The dataset maps a symptom description to a diagnosis.

## 2. English quotes
The english quotes dataset is avaliable [here](https://huggingface.co/datasets/Abirate/english_quotes). 

It contains quotes from famous people in English and the corresponding author.

## 3. Yelp reviews
The yelp reviews dataset is avaliable [here](https://huggingface.co/datasets/codyburker/yelp_review_sampled)

The dataset has reviews written on Yelp and the corresponding 1 to 5 star review.
