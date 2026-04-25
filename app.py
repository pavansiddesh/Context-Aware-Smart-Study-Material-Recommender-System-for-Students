import streamlit as st
import pandas as pd
from src.model import Recommender

# Load data
data = pd.read_csv('data/dataset.csv')

model = Recommender(data)
model.preprocess()
model.compute_similarity()

st.title("📚 Smart Study Recommender")

subject = st.selectbox("Select Subject", data['subject'].unique())
level = st.selectbox("Select Level", data['level'].unique())
type_ = st.selectbox("Select Type", data['type'].unique())

if st.button("Recommend"):
    results = model.recommend(subject, level, type_)
    st.write("### Recommended Materials:")
    st.dataframe(results)
