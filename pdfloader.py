from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documentloaders.py/deepagent.pdf")

docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0])