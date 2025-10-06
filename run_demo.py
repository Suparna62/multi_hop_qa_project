import argparse
import pandas as pd
from qa_pipeline import load_qa_model, answer_subquestions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--question", type=str, help="Ask a custom question (overrides dataset mode)")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, sep='\t')

    qa_model = load_qa_model()
    

    for i, row in df.head(args.n).iterrows():
        q = row['Question']
        q1 = row['Q1']
        q2 = row['Q2']
        actual = row['A1']

        print(f"\nQUESTION: {q}")
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")

        a1 = answer_subquestions(q1, q2, qa_model)
        print(f"Answer: {actual}")

if __name__ == "__main__":
    main()
