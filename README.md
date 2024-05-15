To run embedding and visualisation:
1. Install elasticsearch and add your password to properties.json
2. Create a text file, where each line is a document string to be indexed
3. Add the path to the text file in properties.json as document_file_path
4. Add the path to the text file that the embeddings will be stored in as embeddings_file_path in properties.json
5. Initialize elasticsearch
6. Run embedding procedure and then indexing procedure


To run app:
pip install streamlit transformers elasticsearch umap-learn scikit-learn pandas
streamlit run app/app.py --server.port 8502
