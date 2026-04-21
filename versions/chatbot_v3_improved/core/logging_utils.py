import csv
from datetime import datetime
from core.config import INTERACTION_LOG, FAILED_LOG, FEEDBACK_LOG

def log_interaction(question, answer, response_time, success=1, source=""):
    with open(INTERACTION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "question", "answer", "response_time", "success", "source"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            response_time,
            success,
            source
        ])

def log_failed_query(question, answer, reason="", source=""):
    with open(FAILED_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "question", "answer", "reason", "source"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            reason,
            source
        ])

def log_feedback(question, answer, feedback, source=""):
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "question", "answer", "feedback", "source"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            feedback,
            source
        ])