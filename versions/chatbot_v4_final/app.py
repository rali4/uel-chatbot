import csv
import io
import json
import re
import time
from datetime import datetime

import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from core.config import COLLECTION, DB_DIR, OLLAMA_MODEL
from core.llm import ollama
from core.logging_utils import log_failed_query, log_feedback, log_interaction
from core.prompts import build_prompt
from core.text_utils import clean_output
from core.ui import load_css

st.set_page_config(page_title="UEL Student Support Chatbot v4", layout="wide")
st.markdown(load_css("styles/main.css"), unsafe_allow_html=True)

st.title("UEL Student Support Chatbot v4")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION)


FEE_KEYWORDS = {
    "fee",
    "fees",
    "tuition",
    "refund",
    "refunds",
    "deposit",
    "deposits",
    "sponsor",
    "sponsorship",
    "scholarship",
    "scholarships",
    "bursary",
    "bursaries",
    "payment",
    "payments",
    "instalment",
    "instalments",
    "liability",
    "liable",
    "withdraw",
    "withdrawal",
    "intermission",
    "break",
    "breaks",
    "self-funding",
    "self funding",
    "funding",
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
        items.append({"doc": doc, "meta": meta, "distance": distance})

    return items


def rerank_results(results, fee_query: bool, answer_context_size: int):
    ranked = sorted(
        results,
        key=lambda item: (
            source_priority(item["meta"], fee_query),
            item["distance"] if item["distance"] is not None else 9999,
        ),
    )

    if fee_query:
        policy_chunks = [r for r in ranked if get_source_type(r["meta"]) == "policy"][: max(2, answer_context_size - 2)]
        support_chunks = [r for r in ranked if get_source_type(r["meta"]) == "support_doc"][:2]
        other_chunks = [r for r in ranked if get_source_type(r["meta"]) == "other"][:1]
        final = policy_chunks + support_chunks + other_chunks
        return final[:answer_context_size]

    return ranked[:answer_context_size]


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
        block += f"Distance: {item.get('distance')}\n"
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
        parts.append(
            "SUPPORT / STUDENT-FACING CONTEXT (use only if it does not contradict policy):\n\n"
            + "\n\n---\n\n".join(support_chunks)
        )

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


def build_citation_list(results):
    sources = []
    for idx, item in enumerate(results, start=1):
        meta = item["meta"]
        distance = item.get("distance")
        section = meta.get("section", "-")
        source = meta.get("source_file", "Unknown source")
        snippet = item.get("doc", "")[:180].strip().replace("\n", " ")
        sources.append({
            "id": idx,
            "source": source,
            "section": section,
            "distance": distance,
            "snippet": snippet,
        })
    return sources


def export_chat_json(chat):
    payload = {
        "exported_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "messages": chat,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_chat_csv(chat):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["role", "content", "response_time", "sources"])

    for msg in chat:
        writer.writerow(
            [
                msg.get("role", ""),
                msg.get("content", ""),
                msg.get("response_time", ""),
                " | ".join(msg.get("sources", [])) if msg.get("sources") else "",
            ]
        )

    return output.getvalue()


def answer_confidence(answer: str, results: list) -> str:
    if not results:
        return "Low"

    a = answer.lower()
    if "cannot confirm this from the provided uel documents" in a or "i’m not fully sure" in a or "i'm not fully sure" in a:
        return "Low"

    first_distance = results[0].get("distance")
    if first_distance is None:
        return "Medium"

    if first_distance < 0.25:
        return "High"
    if first_distance < 0.45:
        return "Medium"
    return "Low"


def init_state():
    defaults = {
        "chat": [],
        "last_interaction": None,
        "feedback_given": {},
        "last_retrieval_debug": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

startup_error = None
try:
    embedder = load_embedder()
    col = load_collection()
except Exception as ex:
    embedder = None
    col = None
    startup_error = str(ex)

with st.sidebar:
    st.markdown("## Chatbot Info")
    st.write("Uses official UEL documents to answer student questions.")
    st.write("Prioritises official policy for fee-related questions.")

    st.markdown("## Assistant Controls")
    retrieval_k = st.slider("Retrieval depth", min_value=4, max_value=16, value=8, step=1)
    answer_context_size = st.slider("Chunks used in final answer", min_value=3, max_value=10, value=5, step=1)
    temperature = st.slider("Model temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    show_debug = st.toggle("Show retrieval debug", value=False)

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
        st.session_state.last_retrieval_debug = []
        st.rerun()

    st.markdown("## Export Chat")
    st.download_button(
        "Download JSON",
        data=export_chat_json(st.session_state.chat),
        file_name=f"uel_chat_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        disabled=not st.session_state.chat,
    )
    st.download_button(
        "Download CSV",
        data=export_chat_csv(st.session_state.chat),
        file_name=f"uel_chat_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        disabled=not st.session_state.chat,
    )

    st.markdown("## System Status")
    if startup_error:
        st.error("Knowledge base or model failed to load. Check local setup.")
    else:
        st.success("Retriever and embedder loaded.")

if startup_error:
    st.error(
        "Chatbot startup failed. Confirm your embedding model and Chroma collection exist, then rerun. "
        f"Error: {startup_error}"
    )
    st.stop()

quick_prompts = [
    "When do I become liable for tuition fees?",
    "How can I request extenuating circumstances?",
    "What student support services are available?",
]

st.markdown("### Quick Start")
qp_cols = st.columns(len(quick_prompts))
for i, prompt in enumerate(quick_prompts):
    with qp_cols[i]:
        if st.button(prompt, key=f"quick_{i}"):
            st.session_state["pending_user_question"] = prompt

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            if message.get("sources"):
                st.markdown('<div class="source-box"><b>Sources used</b>', unsafe_allow_html=True)
                for s in message["sources"]:
                    st.write(f"- {s}")
                st.markdown("</div>", unsafe_allow_html=True)

            if message.get("confidence"):
                st.caption(f"Confidence: {message['confidence']}")
            if message.get("response_time") is not None:
                st.caption(f"Response time: {message['response_time']} seconds")

q = st.chat_input("Ask me anything about enrolment, fees, or university policies...")
if not q and st.session_state.get("pending_user_question"):
    q = st.session_state.pop("pending_user_question")

if q:
    user_message = {"role": "user", "content": q}
    st.session_state.chat.append(user_message)

    with st.chat_message("user"):
        st.write(q)

    fee_query = is_fee_question(q)

    retrieval_error = None
    try:
        retrieved = retrieve(q, k=retrieval_k)
    except Exception as ex:
        retrieved = []
        retrieval_error = str(ex)

    ranked_results = rerank_results(retrieved, fee_query, answer_context_size=answer_context_size)
    st.session_state.last_retrieval_debug = build_citation_list(ranked_results)

    if ranked_results:
        context = build_structured_context(ranked_results, fee_query)
        sources = unique_sources(ranked_results)
    else:
        context = ""
        sources = ["No matching source found"]

    source_text = " | ".join(sources)

    if retrieval_error:
        answer = (
            "I’m having trouble searching the knowledge base right now. "
            "Please try again in a moment or contact the Student Hub for urgent support."
        )
        response_time = 0
        success = 0
        log_failed_query(q, answer, f"Retrieval error: {retrieval_error}", source_text)
    else:
        prompt = build_prompt(context=context, question=q, fee_query=fee_query)
        start_time = time.time()
        try:
            raw_answer = ollama(
                prompt,
                model=OLLAMA_MODEL,
                temperature=temperature,
                top_p=top_p,
            )
            answer = clean_output(raw_answer)
        except requests.RequestException as ex:
            answer = (
                "I can’t reach the local language model right now. "
                "Please check if Ollama is running, then try again."
            )
            log_failed_query(q, answer, f"LLM request error: {ex}", source_text)
        except Exception as ex:
            answer = (
                "Something unexpected happened while generating a response. "
                "Please try your question again."
            )
            log_failed_query(q, answer, f"LLM unknown error: {ex}", source_text)

        response_time = round(time.time() - start_time, 2)
        success = 1

        answer_lower = answer.lower()
        if (
            "i cannot confirm this from the provided uel documents" in answer_lower
            or "i'm not fully sure based on the information i have" in answer_lower
            or "i’m not fully sure based on the information i have" in answer_lower
            or (not ranked_results)
        ):
            success = 0
            log_failed_query(q, answer, "Automatic failure flag", source_text)

    confidence = answer_confidence(answer, ranked_results)

    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown('<div class="source-box"><b>Sources used</b>', unsafe_allow_html=True)
        for s in sources:
            st.write(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"Confidence: {confidence}")
        st.caption(f"Response time: {response_time} seconds")

    log_interaction(q, answer, response_time, success, source_text)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "response_time": response_time,
        "confidence": confidence,
    }
    st.session_state.chat.append(assistant_message)

    st.session_state.last_interaction = {
        "question": q,
        "answer": answer,
        "source": source_text,
        "id": len(st.session_state.chat),
    }

if show_debug and st.session_state.last_retrieval_debug:
    st.markdown("### Retrieval Debug")
    for item in st.session_state.last_retrieval_debug:
        with st.expander(
            f"[{item['id']}] {item['source']} | section={item['section']} | distance={item['distance']}",
            expanded=False,
        ):
            st.write(item["snippet"])

last = st.session_state.last_interaction
if last:
    feedback_id = str(last["id"])
    st.markdown("### Feedback")

    if feedback_id not in st.session_state.feedback_given:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Helpful", key=f"helpful_{feedback_id}"):
                log_feedback(last["question"], last["answer"], "Helpful", last["source"])
                st.session_state.feedback_given[feedback_id] = "Helpful"
                st.success("Feedback saved.")

        with col2:
            if st.button("Not helpful", key=f"not_helpful_{feedback_id}"):
                log_feedback(last["question"], last["answer"], "Not helpful", last["source"])
                log_failed_query(
                    last["question"],
                    last["answer"],
                    "User marked as not helpful",
                    last["source"],
                )
                st.session_state.feedback_given[feedback_id] = "Not helpful"
                st.warning("Feedback saved.")
    else:
        st.info(f"Feedback recorded: {st.session_state.feedback_given[feedback_id]}")