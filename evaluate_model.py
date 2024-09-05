from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from datasets import load_dataset
from seqeval.metrics import classification_report

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./model_output")
model = AutoModelForTokenClassification.from_pretrained("./model_output")

# Load the test dataset
test_dataset = load_dataset('json', data_files='dataset/test_transformed.json')['train']

# Define the label map
label_map = {0: "B-QUANTITY", 1: "B-UNIT", 2: "B-FOOD"}

true_labels = []
predicted_labels = []

for example in test_dataset:
    # Tokenize the sentence
    inputs = tokenizer(example['sentence'], return_tensors="pt")
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted labels
    logits = outputs.logits
    predicted_label_ids = torch.argmax(logits, dim=2)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels_example = [label_map[label_id.item()] for label_id in predicted_label_ids[0]]
    
    # Extract the true labels for the tokens
    true_labels_example = [label['label'] for label in example['labels']]
    
    # Align predicted labels with true labels, skipping special tokens
    aligned_true_labels = []
    aligned_predicted_labels = []
    for token, true_label in zip(tokens, true_labels_example):
        if token.startswith("##") or token in ["[CLS]", "[SEP]"]:
            continue
        aligned_true_labels.append(true_label)
        aligned_predicted_labels.append(predicted_labels_example.pop(0))
    
    true_labels.append(aligned_true_labels)
    predicted_labels.append(aligned_predicted_labels)

# Generate a classification report
print(classification_report(true_labels, predicted_labels))