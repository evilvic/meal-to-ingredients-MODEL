import json

def transform_to_simple_format(data):
    transformed_data = []

    for item in data:
        sentence = item['ingredients']
        entities = item['ingredients_entities']

        labels = []
        for entity in entities:
            word = sentence[entity['start']:entity['end']]
            label_type = entity['type']

            # Map label_type to B-QUANTITY, B-UNIT, B-FOOD
            if label_type == 0:
                label = "B-QUANTITY"
            elif label_type == 1:
                label = "B-UNIT"
            elif label_type == 2:
                label = "B-FOOD"
            else:
                label = "O"  # Outside of the main categories

            labels.append({"word": word, "label": label})

        transformed_data.append({"sentence": sentence, "labels": labels})

    return transformed_data

# Lista de archivos de conjuntos de datos para transformar
datasets = ['train', 'valid', 'test']

for dataset in datasets:
    # Cargar el conjunto de datos corregido
    with open(f'dataset/{dataset}_fixed.json', 'r') as file:
        data = json.load(file)

    # Transformar el conjunto de datos
    transformed_data = transform_to_simple_format(data)

    # Guardar el conjunto de datos transformado en un archivo JSON
    with open(f'dataset/{dataset}_transformed.json', 'w') as file:
        json.dump(transformed_data, file, indent=2)

print("Los conjuntos de datos han sido transformados y guardados.")