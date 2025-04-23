import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import streamlit as st

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
CSV_PATH = "assessments.csv"
PICKLE_PATH = "assessments_with_emb.pkl"
EMB_COL = "embedding"

@st.cache_data
def load_data():
    if os.path.exists(PICKLE_PATH):
        df = pd.read_pickle(PICKLE_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
        df[EMB_COL] = df.apply(lambda r: get_embedding(f"{r['name']}. {r['description']}"), axis=1)
        df.to_pickle(PICKLE_PATH)
    return df

def get_embedding(text: str) -> np.ndarray:
    resp = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return np.array(resp["data"][0]["embedding"], dtype=np.float32)

@st.cache_data
def find_best_match(query: str, df: pd.DataFrame, top_k: int = 1):
    q_emb = get_embedding(query).reshape(1, -1)
    all_emb = np.vstack(df[EMB_COL].values)
    sims = cosine_similarity(q_emb, all_emb)[0]
    df2 = df.copy()
    df2["similarity"] = sims
    return df2.nlargest(top_k, "similarity")

def main():
    st.title("Assessment Recommender")
    st.write("Enter your assessment requirements to get the best matching URL.")
    df = load_data()
    query = st.text_input("Enter your query:", "")
    
    if st.button("Get Recommendation") and query:
        with st.spinner("Fetching recommendation..."):
            best = find_best_match(query, df, top_k=1).iloc[0]
            st.markdown(f"**Name:** {best['name']}")
            st.markdown(f"**URL:** [{best['url']}]({best['url']})")
            if 'duration_text' in best:
                st.markdown(f"**Duration:** {best['duration_text']}")
            if 'test_type' in best:
                st.markdown(f"**Test Type:** {best['test_type']}")
            st.markdown(f"**Similarity Score:** {best['similarity']:.4f}")

if __name__ == "__main__":
    main()

