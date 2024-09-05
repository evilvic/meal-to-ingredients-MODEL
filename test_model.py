from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json

# Load the model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./model_output")
tokenizer = AutoTokenizer.from_pretrained("./model_output")

# Load the test dataset
with open('dataset/test_transformed.json', 'r') as f:
    test_data = json.load(f)

label_map = {0: "B-QUANTITY", 1: "B-UNIT", 2: "B-FOOD"}

# Function to predict labels for a sentence
def predict_labels(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted labels
    logits = outputs.logits
    predicted_label_ids = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [label_map[label_id.item()] for label_id in predicted_label_ids[0]]

    return list(zip(tokens, predicted_labels))

# Process each sentence in the test set
for item in test_data:
    sentence = item['sentence']
    predictions = predict_labels(sentence)

    print(f"Sentence: {sentence}")
    print("Predictions:")
    for token, label in predictions:
        print(f"{token}: {label}")
    print("\n" + "-"*50 + "\n")