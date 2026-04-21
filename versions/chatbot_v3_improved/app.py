import re
import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from core.config import DB_DIR, COLLECTION
from core.text_utils import clean_output
from core.llm import ollama
from core.prompts import build_prompt
from core.ui import load_css

st.set_page_config(page_title="UEL Student Support Chatbot V3", layout="wide")
st.markdown(load_css("styles/main.css"), unsafe_allow_html=True)

st.title("UEL Student Support Chatbot")
st.markdown("""
<div class="subtitle">
    Improved prototype with grounded retrieval, cleaner answers, and source visibility.
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION)


embedder = load_embedder()
col = load_collection()

FEE_KEYWORDS = {
    "fee", "fees", "tuition", "refund", "refunds", "deposit", "deposits",
    "sponsor", "sponsorship", "scholarship", "scholarships", "bursary",
    "bursaries", "payment", "payments", "instalment", "instalments",
    "liability", "liable", "withdraw", "withdrawal", "intermission",
    "break", "breaks", "self-funding", "self funding", "funding"
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_fee_question(question: str) -> bool:
    q = normalize_text(question)
    return any(keyword in q for keyword in FEE_KEYWORDS)


def get_source_type(meta: dict) -> str:
    source_file = meta.get("source_file", "").lower()

    if "tuition-fees-policy" in source_file:
        return "policy"
    if "chatbot document" in source_file:
        return "support_doc"
    return "other"


def source_priority(meta: dict, fee_query: bool) -> int:
    source_type = get_source_type(meta)

    if fee_query:
        if source_type == "policy":
            return 0
        if source_type == "support_doc":
            return 1
        return 2

    if source_type == "support_doc":
        return 0
    if source_type == "policy":
        return 1
    return 2


def retrieve(query: str, k: int = 8):
    qemb = embedder.encode([query]).tolist()[0]
    res = col.query(query_embeddings=[qemb], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

    if not docs or not metas:
        return []

    items = []
    for doc, meta, distance in zip(docs, metas, distances):
        items.append({
            "doc": doc,
            "meta": meta,
            "distance": distance,
        })

    return items


def rerank_results(results, fee_query: bool):
    ranked = sorted(
        results,
        key=lambda item: (
            source_priority(item["meta"], fee_query),
            item["distance"] if item["distance"] is not None else 9999
        )
    )

    if fee_query:
        policy_chunks = [r for r in ranked if get_source_type(r["meta"]) == "policy"][:4]
        support_chunks = [r for r in ranked if get_source_type(r["meta"]) == "support_doc"][:2]
        other_chunks = [r for r in ranked if get_source_type(r["meta"]) == "other"][:1]
        final = policy_chunks + support_chunks + other_chunks
    else:
        final = ranked[:5]

    return final


def build_structured_context(results, fee_query: bool) -> str:
    if not results:
        return ""

    policy_chunks = []
    support_chunks = []
    other_chunks = []

    for item in results:
        doc = clean_output(item["doc"])
        meta = item["meta"]
        source_file = meta.get("source_file", "Unknown source")
        section = meta.get("section", "")
        source_type = get_source_type(meta)

        block = f"Source file: {source_file}\n"
        if section:
            block += f"Section: {section}\n"
        block += f"Content:\n{doc}"

        if source_type == "policy":
            policy_chunks.append(block)
        elif source_type == "support_doc":
            support_chunks.append(block)
        else:
            other_chunks.append(block)

    parts = []

    if fee_query and policy_chunks:
        parts.append("OFFICIAL POLICY CONTEXT (highest priority):\n\n" + "\n\n---\n\n".join(policy_chunks))

    if support_chunks:
        parts.append("SUPPORT / STUDENT-FACING CONTEXT:\n\n" + "\n\n---\n\n".join(support_chunks))

    if not fee_query and policy_chunks:
        parts.append("ADDITIONAL POLICY CONTEXT:\n\n" + "\n\n---\n\n".join(policy_chunks))

    if other_chunks:
        parts.append("OTHER CONTEXT:\n\n" + "\n\n---\n\n".join(other_chunks))

    return "\n\n====================\n\n".join(parts)


def unique_sources(results):
    seen = []
    for item in results:
        source_file = item["meta"].get("source_file", "Unknown source")
        if source_file not in seen:
            seen.append(source_file)
    return seen


q = st.text_input("Enter your question:")

if st.button("Ask"):
    if q.strip():
        fee_query = is_fee_question(q)

        with st.spinner("Searching documents..."):
            retrieved = retrieve(q, k=8)
            ranked_results = rerank_results(retrieved, fee_query)

        if ranked_results:
            context = build_structured_context(ranked_results, fee_query)
            sources = unique_sources(ranked_results)
        else:
            context = ""
            sources = ["No matching source found"]

        prompt = build_prompt(context=context, question=q, fee_query=fee_query)

        start_time = time.time()
        with st.spinner("Generating response..."):
            raw_answer = ollama(prompt)
        answer = clean_output(raw_answer)
        response_time = round(time.time() - start_time, 2)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="source-box"><strong>Sources used</strong><br>' +
            "<br>".join([f"• {s}" for s in sources]) +
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(f"Response time: {response_time} seconds")
    else:
        st.warning("Enter a question")