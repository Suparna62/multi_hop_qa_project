import re
import torch
from transformers import pipeline
from ddgs import DDGS  # new library instead of duckduckgo_search

# Load QA model
def load_qa_model():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    print("Loaded QA model: deepset/roberta-base-squad2")
    return qa_pipeline

# Function to answer subquestions
def answer_subquestions(q1, q2, qa_model=None):
    def search_answer(question):
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=5))
        if not results:
            return "No answer found"
        context = " ".join([r["body"] for r in results if "body" in r])
        try:
            result = qa_model(question=question, context=context)
            return result["answer"]
        except Exception:
            return "No answer found"

    a1 = search_answer(q1)
    a2 = search_answer(q2)
    return a1, a2
