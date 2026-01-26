import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "resumes"

def main():
    # Load documents
    documents = SimpleDirectoryReader(
        DATA_DIR,
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()

    # Chunking
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    # Embeddings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Persistent Chroma client
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Build index
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    print(f"Ingested {len(documents)} resume(s) into ChromaDB")

if __name__ == "__main__":
    main()