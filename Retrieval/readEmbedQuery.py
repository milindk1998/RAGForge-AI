from pathlib import Path
import os
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# Allow running this file directly: `python readEmbedQuery.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from Indexing.docparse import load_pdf_documents
from Indexing.docChunk import chunk_documents
from Indexing.docEmbed import embed_documents
from Indexing.docchroma import create_vectorstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").strip().lower()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "genai-rag-eval").strip()
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT


@traceable(name="read_embed_query")
def read_embed_query(pdf_path=None):
    """Load vector context and answer a query using retrieval + ChatGroq."""
    if pdf_path is None:
        pdf_path = str(Path(__file__).resolve().parents[1] / "document" / "companyPolicy.pdf")

    documents = load_pdf_documents(pdf_path)
    chunks = chunk_documents(documents)
    embeddings = embed_documents(chunks)
    vector_store = create_vectorstore(chunks, embeddings)

    # user query
    print("-" * 20)
    print("Welcome to our company portal. We are happy to assist you!")
    query = str(input("\nPlease enter your query below ?\n"))

    # Embed the query and perform a similarity search
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=1)

    # Extract the context from the results and format the prompt
    context = "\n\n".join(document.page_content for document in results)
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using provided context

        If the context is not relevant to the question, say "I don't know".

        Context: {context}
        
        Question: {question}"""
    )
    messages = prompt.format_messages(context=context, question=query)

    # Use the Groq LLM to generate a response based on the context and query
    if not GROQ_API_KEY:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to your environment or .env file."
        )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0.2,
        )
    response = llm.invoke(
        messages,
        config={"run_name": "groq_generation", "tags": ["retrieval", "generation"]},
    )
    return response.content

if __name__ == "__main__":
    response = read_embed_query()
    print("-" * 20)
    print("\nHere is your perfect response from AI:")
    print("\n")
    print(response)
    

