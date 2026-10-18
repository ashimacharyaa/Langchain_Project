import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#loading gemini api
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature = 0
)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name = "medical_research",
    embedding_function = embeddings,
    persist_directory="./chroma_db"
)

def add_pdf(uploaded_file):
    ''' Read a PDF and save its text into ChromaDB.'''

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name

    try:
        reader = PdfReader(temp_path)
        documents= []

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content = text,
                        metadata ={
                            "source":Path(uploaded_file.name).name,
                            "page":page_no
                        }
                    )
                )
        if not documents:
            return 0    

        #split large pages into smaller chunks 
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        #adding searchable text in the document
        db.add_documents(chunks)

        return len(chunks)

    finally:
        os.remove(temp_path)

def retrieve_evidence(question):
    """Find relevant passage from the question"""

    docs = db.similarity_search(question, k=5)

    evidence = []


    for i, doc in enumerate(docs, start=1):
        evidence.append({
            "id":f"S{i}",
            "source":doc.metadata.get("source","Unknown"),
            "page":doc.metadata.get("page","?"),
            "text": doc.page_content
        })
    return evidence

def format_evidence(evidence):
    """Convert retrieve passage into numbered sources."""

    return "\n\n".join(
        f"[{item['id']}]"
        f"{item['source']}, page{item['page']}\n"
        f"{item['text']}"
        for item in evidence
    )

def generate_answer(question, evidence):
    ''' Generate answer grounded in retrieved evidence'''

    prompt = ChatPromptTemplate.from_message([
        (
            "system",
            """ You are a medical research assistant.

Use ONLY the supplied evidence.
Do not invent medical facts.
Do not give personalized diagnosis or treatment.
If evidence is insufficient, say so.

Cite evidence using [S1], [S2], etc.
Do not cite sources that were not provided.

Evidence:
{context}"""
        ),
        ("human","{question}")
    ])

    chain = prompt | llm

    result = chain.invoke({
        "question":question,
        "context":format_evidence(evidence)
    })

    return result.content




