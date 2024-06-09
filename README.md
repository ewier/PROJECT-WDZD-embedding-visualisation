To run embedding and visualisation:
1. Install elasticsearch
2. Modify properties.json - add your elasticsearch password and update the document paths to match your environment
5. Initialize elasticsearch
6. Run embedding procedure and then indexing procedure


To run app: \
pip install streamlit transformers elasticsearch umap-learn scikit-learn pandas \
streamlit run app/app.py --server.port 8502

Dataset source:
https://huggingface.co/datasets/gretelai/symptom_to_diagnosis
