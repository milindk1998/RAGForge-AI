from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable

@traceable(name="load_pdf_documents")
def load_pdf_documents(pdf_path="../document/companyPolicy.pdf"):
    """Load documents from a PDF file and return them."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    documents = load_pdf_documents()
    print("\n")
    print("Number of pages", len(documents))
    print("\n")
    print("All page in pdf content:", [doc.page_content for doc in documents])
    print("\n")
    print("All metadata:", [doc.metadata for doc in documents])
    print("\n")