# model.py
# Wrapper to load QA model and a decomposer function.
# If you have a local decomposer at src/decompose_infer.py it will be used,
# otherwise a simple heuristic fallback will be used.

def load_model(model_name="deepset/roberta-base-squad2"):
    try:
        from transformers import pipeline
        # device=-1 ensures CPU; set device=0 for GPU
        qa = pipeline("question-answering", model=model_name, device=-1)
        print(f"Loaded QA model: {model_name}")
        return qa
    except Exception as e:
        print("Warning: couldn't load QA pipeline:", e)
        return None

def decompose_question(question: str):
    # Try to import your trained decomposer (src/decompose_infer.py -> decompose)
    try:
        from decompose_infer import decompose
        q1, q2 = decompose(question)
        return q1, q2
    except Exception as e:
        # fallback heuristic decomposition
        import re
        q = question.strip()
        # Case: "What is the capital of the birthplace of Rumi?"
        m = re.search(r"birthplace of ([\w\s]+)\??", q, re.I)
        if m:
            name = m.group(1).strip()
            q1 = f"Where was {name} born?"
            q2 = f"What is the capital of the country where {name} was born?"
            return q1, q2
        # General fallback: split on ' of '
        if " of " in q:
            parts = q.split(" of ")
            # create a simple q1 asking about subject
            subject = parts[0].replace("What is the capital", "").strip().rstrip("?")
            q1 = f"Where was {subject} born?"
            q2 = "What is the capital of " + parts[-1].strip().rstrip("?") + "?"
            return q1, q2
        # Default: return original as q1 and empty q2
        return q, ""
