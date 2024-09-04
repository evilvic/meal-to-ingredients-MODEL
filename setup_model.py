from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

model_name = "distilbert-base-uncased"

print("Loading model and tokenizer...")
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Setting up the pipeline...")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

test_input = "Omelette with 2 eggs and 100g of cheese"
print(f"Analyzing text: {test_input}")
result = nlp(test_input)

print("Result:")
for entity in result:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']:.4f}")