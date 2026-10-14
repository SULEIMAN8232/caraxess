from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="career_pilot_course_notes",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


def retrieve_course_material(question: str, k: int = 2) -> list:
    """Retrieve course-note passages relevant to a student's question."""

    if not question.strip():
        raise ValueError("Please enter a question.")

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", "Unknown"),
            "start_index": document.metadata.get("start_index"),
        }
        for document in results
    ]


if __name__ == "__main__":
    test_questions = [
        "What is text splitting?",
        "Explain how RAG answers questions using documents.",
        "What are text embeddings?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")

        results = retrieve_course_material(question)

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}:")
            print(f"Source: {result['source']}")
            print(result["content"])