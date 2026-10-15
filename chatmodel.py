from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=2, max_output_tokens=10 )

result =model.invoke("What is the father of Cristiano Ronaldo JR")

print(result.text)
