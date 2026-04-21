import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

PDF_DIR = "data/pdfs"
DB_DIR = "chroma_db"
COLLECTION = "uel_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)

# Reset collection each rebuild (keeps things simple while you test)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.get_or_create_collection(COLLECTION)

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return chunks

pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
if not pdfs:
    raise SystemExit("No PDFs found in data/pdfs. Add PDFs then rerun.")

docs, metas, ids = [], [], []

for pdf in pdfs:
    path = os.path.join(PDF_DIR, pdf)
    print(f"Reading: {pdf}")
    text = extract_pdf_text(path).strip()
    if not text:
        print(f"WARNING: No text extracted from {pdf}.")
        continue

    chunks = chunk_text(text)
    for i, ch in enumerate(chunks):
        docs.append(ch)
        metas.append({"source_file": pdf, "chunk": i})
        ids.append(f"{pdf}_c{i}")

print("Embedding...")
embeddings = model.encode(docs).tolist()

col.upsert(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
print(f"Done. Indexed {len(docs)} chunks from {len(pdfs)} PDFs into {DB_DIR}.")