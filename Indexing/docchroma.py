try:
    from Indexing.docparse import load_pdf_documents
    from Indexing.docChunk import chunk_documents
    from Indexing.docEmbed import embed_documents
except ModuleNotFoundError:
    # Fallback when running this file directly from the Indexing folder.
    from docparse import load_pdf_documents
    from docChunk import chunk_documents
    from docEmbed import embed_documents
from langchain_chroma import Chroma
from langsmith import traceable
from uuid import uuid4

@traceable(name="create_vectorstore")
def create_vectorstore(chunks, embeddings, collection_name="RAG_doc"):
    """Create vector store using Chroma."""
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=uuids)

    return vector_store

if __name__ == "__main__":
    documents = load_pdf_documents()
    chunks = chunk_documents(documents)
    embeddings = embed_documents(chunks)
    vector_store = create_vectorstore(chunks, embeddings)
    print("\n")
    print("Vector store created successfully.")

    # optional and print the vector store data to verify that it has been created successfully
    # collections = vector_store._collection
    # data = collections.get(include=["metadatas", "documents", "embeddings"])
    # print("\n")
    # print("Total number of documents in vector store: ", len(data["ids"]))
    # print("\n")
    # print("Embeddings stored in Chroma : ", data["embeddings"])
