from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv ()

api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    google_api_key=api_key,
    temperature=0
    )

result = llm.invoke("Who is the President of USA")

print(result.content)