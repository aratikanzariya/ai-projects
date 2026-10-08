import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
import pickle
import numpy as np

from google import genai
from sentence_transformers import SentenceTransformer


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="centered"
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )
    st.stop()


# ==================================================
# GEMINI CLIENT
# ==================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==================================================
# LOAD SAVED EMBEDDINGS
# ==================================================

EMBEDDING_FILE = (
    BASE_DIR / "data" / "embeddings.pkl"
)

if not EMBEDDING_FILE.exists():

    st.error(
        "embeddings.pkl not found inside data folder."
    )

    st.stop()


with open(
    EMBEDDING_FILE,
    "rb"
) as file:

    saved_chunks = pickle.load(file)


# ==================================================
# RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_relevant_chunks(
    question,
    top_k=3
):

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    scored_chunks = []

    for chunk in saved_chunks:

        chunk_embedding = np.array(
            chunk["embedding"]
        )

        score = np.dot(
            question_embedding,
            chunk_embedding
        )

        scored_chunks.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk["source"]
        })

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


# ==================================================
# ASK MY DOCS
# ==================================================

def ask_my_docs(question):

    # ----------------------------------------------
    # Step 1: Retrieve relevant chunks
    # ----------------------------------------------

    retrieved_chunks = (
        retrieve_relevant_chunks(
            question,
            top_k=3
        )
    )


    # ----------------------------------------------
    # Step 2: Keep useful chunks
    # ----------------------------------------------

    useful_chunks = [
        chunk
        for chunk in retrieved_chunks
        if chunk["score"] >= 0.30
    ]


    # ----------------------------------------------
    # Step 3: Create context
    # ----------------------------------------------

    context = "\n\n".join(
        chunk["text"]
        for chunk in useful_chunks
    )


    # ----------------------------------------------
    # Step 4: Create prompt
    # ----------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find the answer in the documents."

Answer:
"""


    # ----------------------------------------------
    # Step 5: Gemini
    # ----------------------------------------------

    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )


    # ----------------------------------------------
    # Step 6: Get answer
    # ----------------------------------------------

    answer = response.output_text


    # ----------------------------------------------
    # Step 7: Return answer and sources
    # ----------------------------------------------

    return answer, useful_chunks


# ==================================================
# FRONTEND
# ==================================================

st.title("📚 Ask My Docs")

st.write(
    "Ask questions from your documents using "
    "RAG, embeddings and Gemini."
)


# ==================================================
# PROJECT INFORMATION
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Documents Chunks",
        len(saved_chunks)
    )

with col2:

    st.metric(
        "Top Results",
        3
    )


st.divider()


# ==================================================
# QUESTION INPUT
# ==================================================

question = st.text_input(
    "💬 Ask a question",
    placeholder="Example: What is machine learning?"
)


# ==================================================
# ASK BUTTON
# ==================================================

if st.button(
    "🔍 Ask My Docs",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, sources = ask_my_docs(
                    question
                )

                # ----------------------------------
                # ANSWER
                # ----------------------------------

                st.subheader("🤖 Answer")

                st.success(answer)


                # ----------------------------------
                # SOURCES
                # ----------------------------------

                st.subheader("📄 Sources")

                if sources:

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.write(
                            f"**{i}. {source['source']}**"
                        )

                        st.write(
                            f"Similarity: "
                            f"{source['score']:.3f}"
                        )

                        with st.expander(
                            "View relevant text"
                        ):

                            st.write(
                                source["text"]
                            )

                        st.divider()

                else:

                    st.warning(
                        "No relevant document "
                        "chunks found."
                    )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📚 Ask My Docs")

    st.write(
        "This project uses:"
    )

    st.write(
        "✅ Sentence Transformers"
    )

    st.write(
        "✅ Embeddings"
    )

    st.write(
        "✅ Cosine Similarity"
    )

    st.write(
        "✅ Top-3 Retrieval"
    )

    st.write(
        "✅ Gemini"
    )

    st.write(
        "✅ RAG"
    )

    st.divider()

    st.write(
        f"Loaded chunks: {len(saved_chunks)}"
    )

    st.write(
        "Similarity threshold: 0.30"
    )