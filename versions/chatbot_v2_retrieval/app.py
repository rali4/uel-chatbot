import os
import streamlit as st
import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

from core.text_utils import clean_output
from core.prompts import build_prompt

OLLAMA_MODEL = "mistral"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "..", "data", "chroma_db")
COLLECTION = "uel_docs"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_css():
    css_path = os.path.join(BASE_DIR, "styles", "main.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")


def retrieve_docs(query: str, n_results: int = 1):
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_or_create_collection(name=COLLECTION)

        query_embedding = embed_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return documents, metadatas

    except Exception as e:
        return [f"Retrieval error: {str(e)}"], []


def ollama_response(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        return result.stdout.strip()

    except Exception as e:
        return f"Error: {str(e)}"


st.set_page_config(page_title="UEL Chatbot Prototype V2", layout="centered")
load_css()

st.title("UEL Chatbot")
st.write("Prototype with document retrieval")

user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("Searching documents..."):
            docs, metadatas = retrieve_docs(user_question)

        clean_docs = [clean_output(doc) for doc in docs]
        context = "\n\n".join(clean_docs)
        context = clean_output(context)

        prompt = build_prompt(context, user_question)

        with st.spinner("Generating response..."):
            answer = ollama_response(prompt)

        cleaned_answer = clean_output(answer)
        st.write(cleaned_answer)

    else:
        st.warning("Enter a question")