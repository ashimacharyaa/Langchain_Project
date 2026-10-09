from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')


model = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    google_api_key=api_key,
    temperature=0
    )

while True:
    user_input = input("You :")
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI:", result.type)