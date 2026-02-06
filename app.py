import os
import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "resumes"

st.set_page_config(page_title="Resume Explorer", layout="wide")
st.title("Resume Explorer")

@st.cache_resource
def load_index():
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY not set. Set it in your environment to use gpt-4o-mini."
        )

    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_output_tokens=512,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(collection)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

index = load_index()
query_engine = index.as_query_engine(similarity_top_k=5)

# Candidate list
st.subheader("Candidates")

def get_candidates():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    results = collection.get(include=["metadatas"])

    candidates = set()
    for meta in results["metadatas"]:
        if meta and "file_name" in meta:
            candidates.add(meta["file_name"])

    return sorted(candidates)

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

candidates = get_candidates()

if not candidates:
    st.warning("No candidates found. Did you run ingest.py?")
else:
    BUTTONS_PER_ROW = 4

    for i in range(0, len(candidates), BUTTONS_PER_ROW):
        row = candidates[i : i + BUTTONS_PER_ROW]
        cols = st.columns(len(row))

        for col, candidate in zip(cols, row):
            if col.button(candidate):
                st.session_state.selected_candidate = candidate

st.divider()

# Candidate details
if st.session_state.selected_candidate:
    st.subheader("Candidate profile")

    detail_query = f"""
    Using ONLY the resume from file:
    {st.session_state.selected_candidate}

    Provide:
    - Name
    - Profession
    - Years of experience
    - Key skills
    - Short professional summary

    If name or any of the information is not available in the resume, state that it is not provided.
    Present the information in a clear and concise manner.
    """

    filters = MetadataFilters(
        filters=[MetadataFilter(key="file_name", value=st.session_state.selected_candidate)]
    )
    filtered_query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters
    )

    with st.spinner("Generating candidate summary..."):
        result = filtered_query_engine.query(detail_query)
    st.markdown(result.response)