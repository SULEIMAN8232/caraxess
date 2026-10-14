from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from document_loader import load_and_split_document


# Project paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "sample_notes.txt"
DB_DIR = BASE_DIR / "chroma_db"


def build_vector_store():
    """Load course notes, embed their chunks, and store them in ChromaDB."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Course notes not found: {DATA_FILE}"
        )

    print("1. Loading and splitting course notes...")

    chunks = load_and_split_document(str(DATA_FILE))
    print(f"   Loaded {len(chunks)} chunks.")

    print("\n2. Initializing Ollama embedding model...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("\n3. Creating ChromaDB vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="career_pilot_course_notes",
        persist_directory=str(DB_DIR),
    )

    print(f"\nSuccess! Vector store saved at: {DB_DIR}")
    print(f"Stored {len(chunks)} document chunks.")

    return vector_store


if __name__ == "__main__":
    try:
        store = build_vector_store()

        print("\n4. Testing semantic search...")

        question = "How does RAG answer questions using documents?"

        results = store.similarity_search(question, k=2)

        print(f"\nQuestion: {question}")

        for index, document in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Source: {document.metadata.get('source')}")
            print(document.page_content)

    except Exception as error:
        print(f"\nError: {error}")