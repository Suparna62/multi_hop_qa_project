
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

def preprocess(tokenizer, examples, question_col, q1_col, q2_col, max_input_length=128, max_output_length=64):
    inputs = ["decompose: " + q for q in examples[question_col]]
    targets = [f"{q1} ||| {q2}" for q1, q2 in zip(examples[q1_col], examples[q2_col])]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=max_output_length, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True, help="Path to train TSV")
    parser.add_argument("--validation-file", type=str, default=None, help="Path to validation TSV")
    parser.add_argument("--model-name", type=str, default="t5-small", help="HuggingFace T5 model")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the model")
    parser.add_argument("--num-epochs", type=int, default=3)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    val_df = pd.read_csv(args.validation_file, sep="\t") if args.validation_file else None

    cols = list(train_df.columns)
    question_col = next((c for c in cols if "question" in c.lower()), None)
    q1_col = next((c for c in cols if c.lower() == "q1"), None)
    q2_col = next((c for c in cols if c.lower() == "q2"), None)

    if not question_col or not q1_col or not q2_col:
        raise ValueError(f"Could not detect required columns. Found: {cols}")

    dataset = DatasetDict({"train": Dataset.from_pandas(train_df)})
    if val_df is not None:
        dataset["validation"] = Dataset.from_pandas(val_df)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    tokenized = dataset.map(
        lambda ex: preprocess(tokenizer, ex, question_col, q1_col, q2_col),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if val_df is not None else None
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
