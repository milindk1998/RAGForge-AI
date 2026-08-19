LangSmith Integration
====================

This project is now wired for LangSmith tracing in the retrieval pipeline.

1. Add these variables to `.env`:
        - `LANGSMITH_API_KEY` (required for tracing)
        - `GROQ_API_KEY` (required for LLM call)
2. Optional LangSmith variables:
        - `LANGSMITH_TRACING=true` (default is `true`)
        - `LANGSMITH_PROJECT=genai-rag-eval` (default project)
        - `LANGSMITH_ENDPOINT=https://api.smith.langchain.com` (default endpoint)
3. Run the query script and check traces in LangSmith.

How the app reads config
------------------------

- Tracing key lookup order:
    `LANGSMITH_API_KEY` -> `LANGCHAIN_API_KEY` -> `langsmith_api_key`
- Groq key lookup order:
    `GROQ_API_KEY` -> `groq_api_key`
- If no LangSmith key is found, the script still runs but tracing is disabled.
- If no Groq key is found, the script stops with a clear error.

Example run command from repo root:

`.\.venv\Scripts\python.exe Retrieval\readEmbedQuery.py`

                        
                        
                        rag_project/
                        │
                        ├── documents/
                        │   └── company_policy.pdf
                        │
                        ├── chroma_db/
                        │
                        ├── rag_pipeline.py
                        │
                        └── test/
                            └── eval.py
                        
                        
                        
                        RAG Application
                            │
                            ├── Indexing
                            │     PDF → Parse → Chunk → HuggingFace Embeddings → ChromaDB
                            │
                            └── Query
                                User Query
                                    ↓
                                Retriever
                                    ↓
                                ChromaDB
                                    ↓
                                Top-3 Chunks
                                    ↓
                                Prompt
                                    ↓
                                ChatGroq
                                    ↓
                                Answer
                                    │
                                    ▼
                            test/eval.py
                                    │
                                    ▼
                            DeepEval + Groq Judge
                                    │
                            ┌──────┼──────────────┐
                            ▼      ▼              ▼
                        Faithfulness  Answer      Contextual
                                        Relevancy    Relevancy
                        
                        
                        
                        
                                        INDEXING
                                            │
                        PDF ──→ PyPDFLoader
                                            │
                                            ▼
                                    Documents
                                            │
                                            ▼
                                    Text Splitter
                                            │
                                            ▼
                                        Chunks
                                            │
                                            ▼
                                HuggingFace Embeddings
                                            │
                                            ▼
                                        ChromaDB
                        
                        
                        
                        Retrieval and Generation
                        
                                        RETRIEVAL
                                            │
                                        User Query
                                            │
                                            ▼
                                HuggingFace Embeddings
                                            │
                                            ▼
                                        ChromaDB
                                            │
                                            ▼
                                    Top-K Documents
                                            │
                                            ▼
                                        LLM
                                            │
                                            ▼
                                        Answer



LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT="RAG_Tracing"
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
GROQ_API_KEY=
