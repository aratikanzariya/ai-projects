import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
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
# LOAD API KEY
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Check your .env file.")
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
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==================================================
# LOAD DOCUMENTS
# ==================================================

@st.cache_data
def load_documents():

    documents = []

    docs_folder = BASE_DIR / "docs"

    for file_path in sorted(docs_folder.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


documents = load_documents()


# ==================================================
# SPLIT TEXT INTO CHUNKS
# ==================================================

def split_text(text, chunk_size=20):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


# ==================================================
# CREATE CHUNKS
# ==================================================

@st.cache_data
def create_chunks(documents):

    chunks = []

    for document in documents:

        document_chunks = split_text(
            document["text"],
            chunk_size=20
        )

        for chunk in document_chunks:

            chunks.append({
                "filename": document["filename"],
                "text": chunk
            })

    return chunks


chunks = create_chunks(documents)


# ==================================================
# CREATE EMBEDDINGS
# ==================================================

@st.cache_data
def create_embeddings(texts):

    embeddings = embedding_model.encode(
        texts
    )

    return np.array(embeddings)


if len(chunks) == 0:

    st.warning(
        "No .txt files found inside the docs folder."
    )

    st.stop()


chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

doc_embeddings = create_embeddings(
    chunk_texts
)


# ==================================================
# SEARCH DOCUMENTS
# ==================================================

def search_documents(question):

    question_embedding = np.array(
        embedding_model.encode(question)
    )

    similarities = (
        doc_embeddings @ question_embedding
        /
        (
            np.linalg.norm(
                doc_embeddings,
                axis=1
            )
            *
            np.linalg.norm(
                question_embedding
            )
        )
    )

    best_index = np.argmax(
        similarities
    )

    return (
        chunks[best_index],
        similarities[best_index],
        best_index
    )


# ==================================================
# ASK MY DOCS
# ==================================================

def ask_my_docs(question):

    # Find relevant chunk
    relevant_chunk, score, index = (
        search_documents(question)
    )

    # Context
    context = relevant_chunk["text"]

    # Prompt
    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say:

"I could not find the answer in the documents."

Answer:
"""

    # Gemini
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    answer = response.output_text

    return (
        answer,
        relevant_chunk,
        score
    )


# ==================================================
# FRONTEND
# ==================================================

st.title("📚 Ask My Docs")

st.write(
    "Ask questions about your documents using "
    "Embeddings + RAG + Gemini."
)


# ==================================================
# DOCUMENT INFORMATION
# ==================================================

st.info(
    f"📄 Documents loaded: {len(documents)}"
)

st.info(
    f"🧩 Total chunks: {len(chunks)}"
)


# ==================================================
# QUESTION INPUT
# ==================================================

question = st.text_input(
    "Ask a question about your documents:",
    placeholder="Example: What is machine learning?"
)


# ==================================================
# ASK BUTTON
# ==================================================

if st.button("🔍 Ask My Docs"):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer, source, score = (
                ask_my_docs(question)
            )


        # ------------------------------------------
        # ANSWER
        # ------------------------------------------

        st.subheader("🤖 Answer")

        st.write(answer)


        # ------------------------------------------
        # SOURCE
        # ------------------------------------------

        st.subheader("📄 Source")

        st.write(
            source["filename"]
        )


        # ------------------------------------------
        # RELEVANT CHUNK
        # ------------------------------------------

        with st.expander(
            "View Relevant Document Chunk"
        ):

            st.write(
                source["text"]
            )


        # ------------------------------------------
        # SIMILARITY SCORE
        # ------------------------------------------

        st.subheader(
            "📊 Similarity Score"
        )

        st.write(
            f"{float(score):.4f}"
        )

        st.progress(
            min(max(float(score), 0.0), 1.0)
        )