import os
import re
import chromadb
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

PDF_DIR = "data/pdfs"
DB_DIR = "chroma_db"
COLLECTION = "uel_docs"

# Point to Tesseract directly (works even if PATH isn't set)
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)

# Reset collection each rebuild while testing
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.get_or_create_collection(COLLECTION)

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return chunks

def ocr_pdf(path: str) -> str:
    doc = fitz.open(path)
    out = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render higher resolution for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="eng")
        text = clean_text(text)
        if text:
            out.append(text)
    return "\n".join(out)

pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
if not pdfs:
    raise SystemExit("No PDFs found in data/pdfs. Put PDFs there and rerun.")

docs, metas, ids = [], [], []

for pdf in pdfs:
    path = os.path.join(PDF_DIR, pdf)
    print(f"OCR reading: {pdf}")
    text = ocr_pdf(path)
    if not text:
        print(f"WARNING: OCR produced no text for {pdf}")
        continue

    for i, ch in enumerate(chunk_text(text)):
        docs.append(ch)
        metas.append({"source_file": pdf, "chunk": i})
        ids.append(f"{pdf}_c{i}")

if not docs:
    raise SystemExit("No text extracted from any PDFs. Check Tesseract install path.")

print("Embedding...")
embeddings = model.encode(docs).tolist()
col.upsert(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

print(f"Done. Indexed {len(docs)} chunks from {len(pdfs)} PDFs into {DB_DIR}.")