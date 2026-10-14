from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path: str) -> list:
    """
    Load a PDF or text file and split its content into chunks.
    Returns a list of LangChain Document objects.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    # Load the file.
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. Use PDF or TXT."
        )

    # IMPORTANT: Initialize documents before using them.
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")

    if not documents:
        return []

    # Remove empty pages.
    documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not documents:
        print("The file contains no extractable text.")
        return []

    # Split the documents.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Generated chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    sample_file = (
        Path(__file__).parent / "data" / "sample_notes.txt"
    )

    chunks = load_and_split_document(str(sample_file))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)