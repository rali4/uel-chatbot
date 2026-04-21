import os
import re
import shutil
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma_db"
COLLECTION = "uel_docs"
PDF_FOLDER = "data/pdfs"

model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("�", " ")
    text = text.replace("\x00", " ")

    # remove obvious junk markers
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # remove non printable junk but keep normal punctuation/newlines
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # fix repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # fix repeated blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_text(pdf_path: str) -> str:
    full_text = []
    doc = fitz.open(pdf_path)

    for page in doc:
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)
        if cleaned:
            full_text.append(cleaned)

    doc.close()
    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def rebuild_database():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.create_collection(name=COLLECTION)

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            print(f"Processing {file}...")

            text = extract_text(path)

            if not text.strip():
                print(f"Skipped {file}: no extractable text found.")
                continue

            chunks = chunk_text(text)

            if not chunks:
                print(f"Skipped {file}: no chunks created.")
                continue

            embeddings = model.encode(chunks).tolist()
            ids = [f"{file}_{i}" for i in range(len(chunks))]
            metadatas = [{"source_file": file} for _ in chunks]

            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

            print(f"Added {len(chunks)} chunks from {file}")

    print("All PDFs added successfully.")


if __name__ == "__main__":
    rebuild_database()