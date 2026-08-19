try:
    from Indexing.docparse import load_pdf_documents
    from Indexing.docChunk import chunk_documents
except ModuleNotFoundError:
    # Fallback when running this file directly from the Indexing folder.
    from docparse import load_pdf_documents
    from docChunk import chunk_documents
from langchain_huggingface import HuggingFaceEmbeddings  
from langsmith import traceable

@traceable(name="embed_documents")
def embed_documents(chunks):

    # Embedding Model
    """Embed documents using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    # manually embed the documents and return the vectors and its optional coz its done in vectordb
    # vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    # return [vectors, embeddings]
    return embeddings

if __name__ == "__main__":
    documents = load_pdf_documents()
    chunker = chunk_documents(documents)
    embeddings = embed_documents(chunker)
    print("\n")
    print("Embedding model:", embeddings)

