from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_db_from_pdf():
    print("📄 STEP 1: Loading PDF...")
    loader = PyMuPDFLoader("./Data/insurence.pdf")
    documents = loader.load()

    print("✅ Pages Loaded:", len(documents))

    # 🔍 DEBUG 1: CHECK RAW TEXT
    print("\n🔍 SAMPLE RAW TEXT:")
    for i, doc in enumerate(documents[:2]):
        print(f"\n--- PAGE {i} ---")
        print(doc.page_content[:300])

    # ❌ If this is empty → PDF problem

    print("\n✂️ STEP 2: Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    print("✅ Total Chunks:", len(docs))

    # 🔍 DEBUG 2: CHECK CHUNKS
    print("\n🔍 SAMPLE CHUNKS:")
    for i, doc in enumerate(docs[:2]):
        print(f"\n--- CHUNK {i} ---")
        print(doc.page_content[:300])

    print("\n🧠 STEP 3: Creating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("\n💾 STEP 4: Creating Vector DB...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./pdf_db"
    )
    db.persist()
    print("✅ DB PERSISTED")

    print("✅ DB Created Successfully")

    # 🔍 DEBUG 3: TEST RETRIEVAL IMMEDIATELY
    print("\n🔍 STEP 5: TESTING RETRIEVAL...")
    results = db.similarity_search("ambulance charges", k=3)

    print("\n📄 Retrieved Results:")
    for r in results:
        print("👉", r.page_content[:200])

if __name__ == "__main__":
    create_db_from_pdf()