
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
    try:
        from decompose_infer import decompose
        q1, q2 = decompose(question)
        return q1, q2
    except Exception as e:
        import re
        q = question.strip()
       
        m = re.search(r"birthplace of ([\w\s]+)\??", q, re.I)
        if m:
            name = m.group(1).strip()
            q1 = f"Where was {name} born?"
            q2 = f"What is the capital of the country where {name} was born?"
            return q1, q2
       
        if " of " in q:
            parts = q.split(" of ")
            subject = parts[0].replace("What is the capital", "").strip().rstrip("?")
            q1 = f"Where was {subject} born?"
            q2 = "What is the capital of " + parts[-1].strip().rstrip("?") + "?"
            return q1, q2
        # Default: return original as q1 and empty q2
        return q, ""

