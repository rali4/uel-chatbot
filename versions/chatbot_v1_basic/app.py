import streamlit as st
import subprocess

OLLAMA_MODEL = "mistral"


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


st.set_page_config(page_title="UEL Chatbot Prototype V1")

st.title("UEL Chatbot")

st.write("Basic chatbot")

user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("Generating..."):
            prompt = f"Answer this question: {user_question}"
            answer = ollama_response(prompt)

        st.write(answer)
    else:
        st.warning("Enter a question")