import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DB_DIR = "data/chroma_db"
COLLECTION = "uel_docs"
PDF_FOLDER = "data/pdfs"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)

try:
    collection = client.get_collection(COLLECTION)
except:
    collection = client.create_collection(COLLECTION)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

doc_id = 0

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)

        print(f"Processing {file}...")

        text = extract_text(path)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()

        ids = [f"{file}_{i}" for i in range(len(chunks))]

        metadatas = [{"source_file": file} for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

print("All PDFs added successfully.")