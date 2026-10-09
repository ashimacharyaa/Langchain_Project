from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model ='gemini-embedding-2', 
    output_dimensionality=32
    )

documents = [
    "Delhi is the capital of India",
    "Kathmandu is the capital of Nepal"
]

result = embeddings.embed_documents(documents)

print(str(result))