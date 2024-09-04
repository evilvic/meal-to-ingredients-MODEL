import json

def fix_entities(data):
    for item in data:
        # Convert the ingredients_entities from string to list of dictionaries
        entities = json.loads(item['ingredients_entities'])
        
        # Ensure 'start' and 'end' indices are integers
        for entity in entities:
            entity['start'] = int(entity['start'])
            entity['end'] = int(entity['end'])

        # Save the corrected entities back to the item
        item['ingredients_entities'] = entities

    return data

# List of dataset files to fix
datasets = ['train', 'valid', 'test']

for dataset in datasets:
    # Load the dataset
    with open(f'dataset/{dataset}.json', 'r') as file:
        data = json.load(file)

    # Fix the entities
    data = fix_entities(data)

    # Save the fixed dataset back to a JSON file
    with open(f'dataset/{dataset}_fixed.json', 'w') as file:
        json.dump(data, file, indent=2)

print("Datasets have been fixed and saved.")