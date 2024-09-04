from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset

# Load the dataset
train_dataset = load_dataset('json', data_files='dataset/train.json')['train']
val_dataset = load_dataset('json', data_files='dataset/valid.json')['train']

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Tokenize the datasets
def tokenize_function(example):
    return tokenizer(example['ingredients'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("./model_output")