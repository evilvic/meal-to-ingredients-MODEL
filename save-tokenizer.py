from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the model and tokenizer from the saved model directory
model = AutoModelForTokenClassification.from_pretrained("./model_output")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Save the tokenizer to the model output directory
tokenizer.save_pretrained("./model_output")

print("Tokenizer saved successfully.")