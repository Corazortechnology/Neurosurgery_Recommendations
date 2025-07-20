from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_handler import QdrantStore

def pdf_to_text(pdf_path: str) -> str:
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

class EmbedDocuments:
    def __init__(self, collection_name: str = "neurosurgery", url: str = "http://localhost:6333"):
        self.qdrant_store = QdrantStore(collection_name=collection_name, url=url)

    def embed_and_store(self, pdf_path: str):
        text = pdf_to_text(pdf_path)
        chunks = split_text(text)
        self.qdrant_store.insert_texts(chunks)

# embedd_docs = EmbedDocuments()
# embedd_docs.embed_and_store("/home/dell-p112f210/Documents/RAG_Chatbot/rag_docs/An_An_Architecture_for_Autism_Concepts_of_Design_I.pdf")

