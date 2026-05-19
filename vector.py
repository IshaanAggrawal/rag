import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FIX 1: Updated pdf_path to match your exact file name 'insurence.pdf'
# FIX 2: Updated db_path to just 'insurance_db' so it saves in your VS Code project
def ensure_insurance_db(pdf_path="data/insurence.pdf", db_path="insurance_db"):
    """Auto-creates the Chroma DB if the PDF exists but the DB doesn't."""
    if not os.path.exists(db_path) and os.path.exists(pdf_path):
        print("📄 Reading insurence.pdf and creating local embeddings...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_path)
        print("✅ Insurance DB created successfully!")

def load_vectorstore(path: str):
    if not os.path.exists(path):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=path, embedding_function=embeddings)

def get_vectorstores():
    ensure_insurance_db() # Check and build Insurance DB if needed
    
    # Original DBs
    heart_db = load_vectorstore("/mnt/efs/chroma_db")
    gyno_db = load_vectorstore("/mnt/efs/gyno_db")
    
    # Load our new local DB
    insurance_db = load_vectorstore("insurance_db")
    
    return {"heart": heart_db, "gyno": gyno_db, "insurance": insurance_db}