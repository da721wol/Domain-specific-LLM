"""REFERENCE MATERIAL: https://huggingface.co/docs/evaluate/transformers_integrations"""
"""https://github.com/toughdata/Flan-T5-Quora-Question-Answering/blob/main/finetune_flan_t5.py"""

# Import the necessary libraries
import nltk
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load and split the dataset
dataset = load_dataset("philschmid/germeval18")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dwolpers/german_T5_Large_Closed")
model = T5ForConditionalGeneration.from_pretrained("dwolpers/german_T5_Large_Closed")

# Load the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

prefix = "Bewerte folgende Aussage: "

def preprocess_function(examples):
  """Add prefix to the sentences, tokenize the text, and set the labels"""
  # The "inputs" are the tokenized answer:
  inputs = [prefix + doc for doc in examples["text"]]
  model_inputs = tokenizer(inputs, max_length=128, truncation=True)

  # The "labels" are the tokenized outputs:
  labels = tokenizer(text_target=examples["multi"], max_length=512, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up Rouge score for evaluation
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
  preds, labels = eval_preds

  # decode preds and labels
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  # rougeLSum expects newline after each sentence
  decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
  decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

  result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  return result

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
  output_dir="dwolpers/german_T5_Large_Closed_Class",
  evaluation_strategy="epoch",
  learning_rate=3e-4,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=8,
  weight_decay=0.01,
  save_total_limit=3,
  num_train_epochs=6,
  predict_with_generate=True,
  push_to_hub=True,
  report_to=["none"]
)

# Set up trainer
trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset["train"],
  eval_dataset=tokenized_dataset["test"],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

# Train the model
trainer.train()