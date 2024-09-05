from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset

# Load the dataset
train_dataset = load_dataset('json', data_files='dataset/train_transformed.json')['train']
val_dataset = load_dataset('json', data_files='dataset/valid_transformed.json')['train']

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=3)  # Adjust num_labels to match your dataset

# Define a function to align labels with tokens
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['sentence'], truncation=True, padding='max_length')

    labels = []
    for i, label_set in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words in original text
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens like [CLS], [SEP], etc.
            elif word_idx != previous_word_idx:  # New word
                if word_idx < len(label_set):
                    label = label_set[word_idx]['label']
                    if label == "B-QUANTITY":
                        label_ids.append(0)
                    elif label == "B-UNIT":
                        label_ids.append(1)
                    elif label == "B-FOOD":
                        label_ids.append(2)
                    else:
                        label_ids.append(-100)  # If label is unknown, ignore it
                else:
                    label_ids.append(-100)  # If word_idx is out of label's range
            else:
                label_ids.append(-100)  # For subsequent subwords of the same word
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# Tokenize and align labels for the dataset
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
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

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save the model
trainer.train()
trainer.save_model("./model_output")