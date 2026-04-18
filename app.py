import re
import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from core.config import DB_DIR, COLLECTION
from core.text_utils import clean_output
from core.logging_utils import log_interaction, log_failed_query, log_feedback
from core.llm import ollama
from core.prompts import build_prompt
from core.ui import load_css

st.set_page_config(page_title="UEL Student Support Chatbot", layout="wide")
st.markdown(load_css("styles/main.css"), unsafe_allow_html=True)

st.title("UEL Student Support Chatbot")

st.markdown("""
<div style="
    background: rgba(255,255,255,0.05);
    padding: 18px 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
">
    <h3 style="margin:0; color:white;">Welcome</h3>
    <p style="margin:8px 0 0 0; color:#cdd6f4;">
        Ask about enrolment, fees, assessments, deadlines, or student support services using official UEL documents.
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Chatbot Info")
    st.write("This chatbot uses official UEL documents to answer student questions.")
    st.write("It does not guess or invent information.")
    st.write("If the answer is not clearly supported by the documents, it will direct you to the Student Hub.")

    st.markdown("## Common Topics")
    st.write("- Enrolment")
    st.write("- Tuition fees")
    st.write("- Assessments")
    st.write("- Extenuating circumstances")
    st.write("- Student support")

    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.session_state.last_interaction = None
        st.session_state.feedback_given = {}
        st.rerun()


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
        doc = item["doc"]
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
        parts.append("SUPPORT / STUDENT-FACING CONTEXT (use only if it does not contradict policy):\n\n" + "\n\n---\n\n".join(support_chunks))

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


if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            if message.get("sources"):
                st.markdown('<div class="source-box"><b>Sources used</b>', unsafe_allow_html=True)
                for s in message["sources"]:
                    st.write(f"- {s}")
                st.markdown("</div>", unsafe_allow_html=True)

            if message.get("response_time") is not None:
                st.caption(f"Response time: {message['response_time']} seconds")


q = st.chat_input("Ask me anything about enrolment, fees, or university policies...")

if q:
    user_message = {
        "role": "user",
        "content": q
    }
    st.session_state.chat.append(user_message)

    with st.chat_message("user"):
        st.write(q)

    fee_query = is_fee_question(q)
    retrieved = retrieve(q, k=8)
    ranked_results = rerank_results(retrieved, fee_query)

    if ranked_results:
        context = build_structured_context(ranked_results, fee_query)
        sources = unique_sources(ranked_results)
    else:
        context = ""
        sources = ["No matching source found"]

    source_text = " | ".join(sources)
    prompt = build_prompt(context=context, question=q, fee_query=fee_query)

    start_time = time.time()
    raw_answer = ollama(prompt)
    answer = clean_output(raw_answer)
    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    success = 1
    answer_lower = answer.lower()

    if (
        "i cannot confirm this from the provided uel documents" in answer_lower
        or "i'm not fully sure based on the information i have" in answer_lower
        or (not ranked_results)
    ):
        success = 0
        log_failed_query(q, answer, "Automatic failure flag", source_text)

    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown('<div class="source-box"><b>Sources used</b>', unsafe_allow_html=True)
        for s in sources:
            st.write(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"Response time: {response_time} seconds")

    log_interaction(q, answer, response_time, success, source_text)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "response_time": response_time
    }
    st.session_state.chat.append(assistant_message)

    st.session_state.last_interaction = {
        "question": q,
        "answer": answer,
        "source": source_text,
        "id": len(st.session_state.chat)
    }


last = st.session_state.last_interaction

if last:
    feedback_id = str(last["id"])

    st.markdown("### Feedback")

    if feedback_id not in st.session_state.feedback_given:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Helpful", key=f"helpful_{feedback_id}"):
                log_feedback(
                    last["question"],
                    last["answer"],
                    "Helpful",
                    last["source"]
                )
                st.session_state.feedback_given[feedback_id] = "Helpful"
                st.success("Feedback saved.")

        with col2:
            if st.button("Not helpful", key=f"not_helpful_{feedback_id}"):
                log_feedback(
                    last["question"],
                    last["answer"],
                    "Not helpful",
                    last["source"]
                )
                log_failed_query(
                    last["question"],
                    last["answer"],
                    "User marked as not helpful",
                    last["source"]
                )
                st.session_state.feedback_given[feedback_id] = "Not helpful"
                st.warning("Feedback saved.")
    else:
        st.info(f"Feedback recorded: {st.session_state.feedback_given[feedback_id]}")