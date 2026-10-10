from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-2',
    output_dimensionality=32
    )

documents = [
    "He is the hero of the whole universe",
    "Wait urmila within a few decades you will see who we are"
]

he = embeddings.embed_documents(documents)

print(he)