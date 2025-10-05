import pandas as pd
import argparse
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset (tsv)')
    parser.add_argument('--model', required=True, help='Path to decomposer model')
    parser.add_argument('--n', type=int, default=None, help='Number of examples to evaluate')
    args = parser.parse_args()


    df = pd.read_csv(args.dataset, sep='\t')
    df.columns = [col.strip() for col in df.columns]

    col_map = {'Question': 'question', 'Q1': 'q1', 'Q2': 'q2', 'A1': 'answer'}
    df.rename(columns=col_map, inplace=True)

    if args.n:
        df = df.head(args.n)

    results = []

    for index, row in df.iterrows():
        question = row['question']
        q1 = row['q1']
        q2 = row['q2']
        answer = row['answer']

        results.append({
            'question': question,
            'q1': q1,
            'q2': q2,
            'answer': answer,
        })

        print({
            'question': question,
            'q1': q1,
            'q2': q2,
            'answer': answer,
        })

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

