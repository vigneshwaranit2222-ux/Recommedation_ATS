import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="AI Recruitment Suite", layout="wide")
st.title("🤖 AI Recruitment & Hiring Suite")

tab1, tab2, tab3 = st.tabs(["1. Generate Job", "2. Interview Chat", "3. Rank Candidates"])

# Tab 1: Generate Job
with tab1:
    st.header("Generate Job Description")
    raw_prompt = st.text_area("Prompt / Requirements", "Need 2 YOE React Developer with Tailwind & GraphQL")
    
    if st.button("Generate & Index Job"):
        with st.spinner("Calling AI Pipeline..."):
            res = requests.post(f"{API_BASE_URL}/jobs/generate", json={"prompt": raw_prompt})
            if res.status_code == 200:
                data = res.json()
                st.success(f"Job Created Successfully! Job ID: {data.get('id')}")
                st.json(data)
            else:
                st.error(f"Error {res.status_code}: {res.text}")

# Tab 2: Candidate Ranking
with tab3:
    st.header("Rank Resumes")
    job_id = st.text_input("Job ID")
    resume_text = st.text_area("Paste Candidate Resume Content")
    candidate_id = st.text_input("Candidate ID", "cand_001")
    
    if st.button("Rank Candidate"):
        payload = {
            "candidates": [
                {"candidate_id": candidate_id, "resume_text": resume_text}
            ]
        }
        res = requests.post(f"{API_BASE_URL}/jobs/{job_id}/rank", json=payload)
        if res.status_code == 200:
            st.write("### Ranking Breakdown")
            st.json(res.json())
        else:
            st.error(f"Error: {res.text}")  