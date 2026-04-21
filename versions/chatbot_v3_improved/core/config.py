import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")

COLLECTION = "uel_docs"

OLLAMA_MODEL = "mistral"

INTERACTION_LOG = os.path.join(PROJECT_ROOT, "data", "interaction_log.csv")

FAILED_LOG = os.path.join(PROJECT_ROOT, "data", "failed_queries.csv")

FEEDBACK_LOG = os.path.join(PROJECT_ROOT, "data", "feedback_log.csv")