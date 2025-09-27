import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

# Load environment variables
QDRANT_URL = os.getenv("AGENT_QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("AGENT_QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("AGENT1_QDRANT_COLLECTION", "jobyaari_jobs")
OLLAMA_HOST = os.getenv("AGENT_OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("AGENT_EMBEDDING_MODEL", "nomic-embed-text:latest")

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=EMBEDDING_MODEL
)

# Wrap Qdrant with LangChain VectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

# Example: simple retrieval
def test_retrieval():
    query_text = "latest job in upsc"
    try:
        results = vector_store.similarity_search(query_text, k=5)
        print("Retrieved results:")
        for r in results:
            print(r)
    except Exception as e:
        print(f"Error retrieving from Qdrant: {e}")

if __name__ == "__main__":
    test_retrieval()
