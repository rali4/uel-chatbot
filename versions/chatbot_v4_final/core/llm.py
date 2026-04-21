import requests
from core.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL

def ollama(prompt, model=OLLAMA_MODEL, temperature=0.2, top_p=0.9):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        },
        timeout=OLLAMA_TIMEOUT
    )

    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()