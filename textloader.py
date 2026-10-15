from langchain_community.document_loaders import TextLoader

loader = TextLoader("documentloaders.py/rag.txt", encoding="utf-8")

documents = loader.load()

print(type(documents))

print(len(documents))

print(documents[0])

print(type(documents[0]))