# shl_recommendation_system.py
import os
import re
import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import streamlit as st
import uvicorn

# Configuration
MODEL_NAME = "bert-base-uncased"
SHL_URL = "https://www.shl.com/solutions/products/product-catalog/"
EMBEDDING_DIM = 768
TEST_QUERIES = [
    "Java developers with collaboration skills, 40 mins",
    "Python/SQL/JS mid-level, 60 mins",
    "Analyst cognitive/personality tests, 45 mins"
]

# Initialize components
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class QueryRequest(BaseModel):
    query: str

def setup_chrome_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service('/usr/bin/chromedriver'), options=options)

def scrape_shl_catalog():
    driver = setup_chrome_driver()
    driver.get(SHL_URL)
    
    # Scroll to load all products
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    assessments = []
    
    for item in soup.select('.product-item'):
        # Parse duration to numerical value
        duration_text = item.select_one('.duration').text.strip()
        duration = int(re.search(r'\d+', duration_text).group()) if 'min' in duration_text.lower() else 0
        
        assessments.append({
            'name': item.select_one('.product-name').text.strip(),
            'url': item.select_one('a')['href'],
            'remote_support': 'Yes' if 'Remote' in item.text else 'No',
            'adaptive_support': 'Yes' if 'Adaptive' in item.text else 'No',
            'duration': duration,
            'duration_text': duration_text,
            'test_type': item.select_one('.test-type').text.strip(),
            'description': item.select_one('.product-description').text.strip()
        })
    
    driver.quit()
    return pd.DataFrame(assessments)

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def preprocess_data():
    if not os.path.exists('shl_data.csv'):
        df = scrape_shl_catalog()
        df.to_csv('shl_data.csv', index=False)
    else:
        df = pd.read_csv('shl_data.csv')
    
    if 'embedding' not in df.columns:
        embeddings = generate_embeddings(df['name'] + " " + df['description'])
        df['embedding'] = list(embeddings)
        df.to_pickle('shl_embeddings.pkl')
    
    return pd.read_pickle('shl_embeddings.pkl')

def parse_query(query):
    # Extract duration constraint
    duration_match = re.search(r'(\d+)(?:-minute|min|minutes)', query, re.IGNORECASE)
    max_duration = int(duration_match.group(1)) if duration_match else None
    
    # Extract skills
    classification = zero_shot_classifier(
        query,
        candidate_labels=["Java", "Python", "SQL", "JavaScript", 
                         "Collaboration", "Cognitive", "Personality",
                         "Analytical", "Technical", "Soft Skills"]
    )
    skills = [label for label, score in zip(classification['labels'], 
             classification['scores']) if score > 0.6]
    
    return {
        'skills': skills,
        'max_duration': max_duration
    }

def recommend_assessments(query, top_k=10):
    df = preprocess_data()
    parsed = parse_query(query)
    
    # Process URL input
    if query.startswith('http'):
        response = requests.get(query)
        soup = BeautifulSoup(response.text, 'html.parser')
        query_text = soup.get_text()
    else:
        query_text = query
    
    # Generate query embedding
    query_embedding = generate_embeddings([query_text])
    sims = cosine_similarity(query_embedding, np.stack(df['embedding'].values))[0]
    df['similarity'] = sims
    
    # Apply filters
    filtered = df
    if parsed['skills']:
        filtered = filtered[
            filtered['name'].str.contains('|'.join(parsed['skills']), case=False) |
            filtered['description'].str.contains('|'.join(parsed['skills']), case=False)
        ]
    if parsed['max_duration']:
        filtered = filtered[filtered['duration'] <= parsed['max_duration']]
    
    # Ensure minimum 1 recommendation
    results = filtered.nlargest(top_k, 'similarity')[[
        'name', 'url', 'remote_support', 'adaptive_support', 
        'duration_text', 'test_type'
    ]].to_dict('records')
    
    return results[:10] if len(results) > 0 else df.nlargest(1, 'similarity')[[
        'name', 'url', 'remote_support', 'adaptive_support', 
        'duration_text', 'test_type'
    ]].to_dict('records')

# FastAPI setup
app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", status_code=status.HTTP_200_OK)
def recommend(request: QueryRequest):
    try:
        results = recommend_assessments(request.query)
        if not results:
            return {"recommendations": []}
        
        # Format response exactly as required
        formatted = []
        for res in results:
            formatted.append({
                "assessment_name": res['name'],
                "assessment_url": res['url'],
                "remote_testing_support": res['remote_support'],
                "adaptive_irt_support": res['adaptive_support'],
                "duration": res['duration_text'],
                "test_type": res['test_type']
            })
        return {"recommendations": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit app
def run_streamlit():
    st.title("SHL Assessment Recommender")
    
    query = st.text_area("Enter job description or query:")
    if st.button("Get Recommendations"):
        with st.spinner("Processing..."):
            response = requests.post(
                "http://localhost:8000/recommend",
                json={"query": query}
            ).json()
            
            if response['recommendations']:
                st.table(response['recommendations'])
            else:
                st.warning("No assessments found matching the criteria")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
