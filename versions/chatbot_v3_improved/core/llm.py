import requests
from core.config import OLLAMA_MODEL

def ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()