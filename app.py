import streamlit as st
from predict import predict_fake_news

st.title("📰 Fake News Classifier")

st.write("Enter the **title** and **text** of the news article:")

title_input = st.text_area("News Title", height=100)
text_input = st.text_area("News Content", height=200)

if st.button("Predict"):
    if not title_input or not text_input:
        st.warning("Please provide both title and text.")
    else:
        label, prob = predict_fake_news(title_input, text_input)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {prob:.4f}")
