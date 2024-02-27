from datasets import load_dataset
import evaluate
import torch

from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load dataset
dataset = load_dataset("deepset/germanquad", split="test")

# Load rouge for validation
rouge = evaluate.load("rouge")


def evaluate_open_book(model, tokenizer, name, dataset=dataset, rouge=rouge):

  def generate_answer(batch):
    inputs = ["Beantworte mit folgendem Kontext die Frage: " + row[0] + " " + row[1] for row in zip(batch['question'], batch['context'])]
    input_dict = tokenizer(inputs, max_length=384, truncation=True, padding=True,  return_tensors="pt")
    input_dict = {k: v.to(model.device) for k, v in input_dict.items()}
    predicted_ids = model.generate(
      input_ids=input_dict['input_ids'],
      attention_mask=input_dict['attention_mask'],
      max_length=300,
      length_penalty=2.0,
      num_beams=4,
      early_stopping=True
    )
    # Convert output tensor to array, then to list and to string.
    batch["predicted_answer"] = [tokenizer.decode(id, skip_special_tokens=True) for id in predicted_ids]
    return batch

  # The model needs to run on a GPU, if you have one.
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  # Generate answers
  dataset = dataset.map(generate_answer, batched=True, batch_size=8)

  # Calculate rouge scores
  answers = dataset["answers"]
  answers = [example["text"][0] for example in answers]
  rouge_output = rouge.compute(predictions=dataset["predicted_answer"], references=answers, rouge_types=["rouge1", "rouge2", "rougeL"])
  rouge_output["model"] = name
  with open('../quad_open_book.jsonl', 'a') as f:
    f.write(str(rouge_output))
    f.write("\n")


# Load a local model and tokenizer
t5_base_open = T5ForConditionalGeneration.from_pretrained("data/dwolpers/german_T5_Base_Open")
t5_base_open_tokenizer = AutoTokenizer.from_pretrained("data/dwolpers/german_T5_Base_Open")

evaluate_open_book(t5_base_open, t5_base_open_tokenizer, "t5_base_open")

t5_base_open_class = T5ForConditionalGeneration.from_pretrained("data/dwolpers/german_T5_Base_Open_Class")
t5_base_open_class_tokenizer = AutoTokenizer.from_pretrained("data/dwolpers/german_T5_Base_Open_Class")

evaluate_open_book(t5_base_open_class, t5_base_open_class_tokenizer, "t5_base_open_class")

t5_large_open = T5ForConditionalGeneration.from_pretrained("data/dwolpers/german_T5_Large_Open")
t5_large_open_tokenizer = AutoTokenizer.from_pretrained("data/dwolpers/german_T5_Large_Open")

evaluate_open_book(t5_large_open, t5_large_open_tokenizer, "t5_large_open")

t5_large_open_class = T5ForConditionalGeneration.from_pretrained("data/dwolpers/german_T5_Large_Open_Class")
t5_large_open_class_tokenizer = AutoTokenizer.from_pretrained("data/dwolpers/german_T5_Large_Open_Class")

evaluate_open_book(t5_large_open_class, t5_large_open_class_tokenizer, "t5_large_open_class")






