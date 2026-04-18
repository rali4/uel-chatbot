import re

def clean_output(text):
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', text)
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', '', text)

    # Fix weird symbols
    text = text.replace("�", "")

    # Fix spacing
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove repeated words (the the, and and)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Fix broken short fragments (like "p page")
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z]{3,})\b', r'\2', text)

    # Fix duplicated URLs
    text = re.sub(r'(https?://\S+)\s+\1', r'\1', text)

    # Fix partial duplicated URLs (like your screenshot)
    text = re.sub(r'(https?://[^\s]+)\s+(https?://[^\s]+)', r'\2', text)

    # Remove isolated numbering like "3." on its own
    text = re.sub(r'\s*\b\d+\.\s*(?=\s|$)', '', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        cleaned.append(s)

    text = " ".join(cleaned)

    # Break into readable lines if long
    if len(text) > 250:
        parts = re.split(r'(?<=[.!?])\s+', text)
        text = "\n\n".join(parts[:6])

    return text