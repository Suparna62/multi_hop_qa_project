
# evaluate.py
import argparse, json, re
import pandas as pd
from decompose_infer import decompose
from qa_pipeline import answer_subquestion
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

def normalize(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s

def compute_f1(a_gold, a_pred):
    g = normalize(a_gold).split()
    p = normalize(a_pred).split()
    if not g or not p: return 0.0
    common = set(g) & set(p)
    if not common: return 0.0
    prec = len(common)/len(p)
    rec = len(common)/len(g)
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, sep='\t')
    if args.n:
        df = df.head(args.n)

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = []
    for _, row in df.iterrows():
        q = row['question']
        ref_q1 = row['q1']
        ref_q2 = row['q2']
        ref_ans = row['answer']

        pred_q1, pred_q2 = decompose(args.model, q)
        # Fill entity if needed: simple insert if q2 contains placeholder
        composed_q2 = pred_q2.replace('<ENTITY>', pred_q1) if '<ENTITY>' in pred_q2 else pred_q2 + " " + pred_q1
        a1, s1 = answer_subquestion(pred_q1)
        a2, s2 = answer_subquestion(composed_q2)
        final = a2 if a2 else a1

        # scores
        rouge1 = rouge.score(ref_q1, pred_q1)['rougeL'].fmeasure
        rouge2 = rouge.score(ref_q2, pred_q2)['rougeL'].fmeasure
        f1 = compute_f1(ref_ans, final)
        em = (normalize(ref_ans) == normalize(final))

        results.append({
            "question": q, "ref_answer": ref_ans, "pred_answer": final,
            "ref_q1": ref_q1, "pred_q1": pred_q1, "ref_q2": ref_q2, "pred_q2": pred_q2,
            "rouge_q1": rouge1, "rouge_q2": rouge2, "final_f1": f1, "final_em": em
        })

    outp = {"results": results}
    print(json.dumps(outp, indent=2))

if __name__ == "__main__":
    main()
