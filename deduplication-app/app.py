import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="AI Deduplication Tool", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def clean_text(s):
    return str(s).strip().lower()

def combine_columns(df, columns):
    return df[columns].fillna('').astype(str).agg(' '.join, axis=1)

def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=False)

def cluster_embeddings(embeddings, eps=0.15, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    return clustering.labels_

st.title("🧠 AI-Powered Data Cleansing & Deduplication Tool")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    selected_columns = st.multiselect("Select columns to deduplicate by (combine for smarter matching)", df.columns)

    if selected_columns:
        threshold = st.slider("Clustering Sensitivity (lower = stricter)", 0.05, 0.5, 0.15, 0.01)

        if st.button("🔍 Run Deduplication"):
            with st.spinner("Generating embeddings and clustering..."):
                combined_text = combine_columns(df, selected_columns)
                cleaned_texts = combined_text.apply(clean_text).tolist()
                embeddings = get_embeddings(cleaned_texts)
                labels = cluster_embeddings(embeddings, eps=threshold)
                df['Duplicate Group'] = labels

                num_groups = len(set(labels)) - (1 if -1 in labels else 0)
                st.success(f"Found {num_groups} duplicate groups")

                grouped = df[df['Duplicate Group'] != -1].sort_values(by='Duplicate Group')
                st.write("### Potential Duplicate Groups")
                st.dataframe(grouped)

                csv = grouped.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Duplicates CSV", csv, "deduplicated_data.csv", "text/csv")