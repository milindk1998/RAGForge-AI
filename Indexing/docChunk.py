try:
    from Indexing.docparse import load_pdf_documents
except ModuleNotFoundError:
    # Fallback when running this file directly from the Indexing folder.
    from docparse import load_pdf_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

@traceable(name="chunk_documents")
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    documents = load_pdf_documents()
    print("\n")
    print("Number of pages", len(documents))
    chunks = chunk_documents(documents)
    print("\n")
    print("Number of chunks:", len(chunks))
    print("\n")
    for i, chunk in enumerate(chunks):
        print("\n")
        print(f"Chunk {i + 1}:")
        print("\n")
        print(chunk.page_content)
        print("\n")
        print("Metadata: ", chunk.metadata)
    print("\n")