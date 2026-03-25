import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

DB_DIR = "chroma_db"
COLLECTION = "uel_docs"
OLLAMA_MODEL = "mistral"

st.set_page_config(page_title="UEL Student Support Chatbot", layout="wide")
st.title("UEL Student Support Chatbot")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION)

embedder = load_embedder()
col = load_collection()

def retrieve(q, k=3):
    qemb = embedder.encode([q]).tolist()[0]
    res = col.query(query_embeddings=[qemb], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))

def ollama(prompt):
    p = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout.decode("utf-8").strip()

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

q = st.chat_input("Ask a question about UEL policies...")

if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.write(q)

    retrieved = retrieve(q)
    context = "\n\n---\n\n".join([d for d, _ in retrieved])
    sources = list({m["source_file"] for _, m in retrieved})

    rules = (
    "You are a Student Support Chatbot for a UK university.\n"
    "You provide clear, accurate, policy-based guidance to students.\n\n"

    "STRICT RULES:\n"
    "1. Use ONLY the provided CONTEXT from official university documents.\n"
    "2. If the answer is not explicitly stated in the CONTEXT, say: "
    "'I cannot confirm this from the provided UEL documents.'\n"
    "3. Do NOT guess, infer, assume, or create information.\n"
    "4. Do NOT invent deadlines, policies, fees, or procedures.\n"
    "5. If the question is outside the documents, advise the student to contact the Student Hub.\n"
    "6. Do not provide legal, financial, medical, or immigration advice beyond what is written in the policy.\n"
    "7. Do not contradict the document wording.\n\n"

    "TONE AND STYLE:\n"
    "8. Use clear, simple language suitable for students.\n"
    "9. Use short sentences.\n"
    "10. Present answers in bullet points.\n"
    "11. Be neutral, professional, and supportive.\n"
    "12. Do not use technical AI language.\n\n"

    "STRUCTURE:\n"
    "13. Start with a direct answer to the question.\n"
    "14. Then provide step-by-step explanation if applicable.\n"
    "15. Where possible, include a short quoted phrase from the policy to support the answer.\n"
    "16. If relevant, explain what the student should do next.\n\n"

    "SAFETY:\n"
    "17. If the student appears distressed or in urgent need, advise them to contact the Student Hub or appropriate university service.\n"
    "18. Do not process personal data.\n"
)

    prompt = f"{rules}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nANSWER:"
    answer = ollama(prompt)

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption("Sources used:")
        for s in sources:
            st.write(f"- {s}")

    st.session_state.chat.append(("assistant", answer))