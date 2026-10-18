from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()



api_key = os.getenv('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    google_api_key=api_key,
    temperature=0
    )

st.header("Research paper summarize tool")

user_input = st.text_input("Enter your queries")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.type)