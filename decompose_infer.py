
# decompose_infer.py
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import argparse

def decompose(model_dir, question, max_length=128):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    input_ids = tokenizer("decompose: " + question, return_tensors="pt", truncation=True).input_ids
    outs = model.generate(input_ids, max_length=max_length, num_beams=4)
    decoded = tokenizer.decode(outs[0], skip_special_tokens=True)
    if "<sep>" in decoded:
        q1, q2 = [s.strip() for s in decoded.split("<sep>")]
    else:
        # fallback simple split
        parts = decoded.split("?")
        if len(parts) >= 2:
            q1 = parts[0].strip()+"?"
            q2 = "?".join(parts[1:]).strip()
        else:
            q1, q2 = decoded, ""
    return q1, q2

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--question", type=str, required=False)
    args = parser.parse_args()
    if args.question:
        q1,q2 = decompose(args.model, args.question)
        print("Q1:", q1)
        print("Q2:", q2)
    else:
        print("Provide --question to infer.")
