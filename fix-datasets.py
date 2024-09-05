import json

def fix_entities(data):
    # Mapeo de los tipos a índices numéricos
    type_to_index = {
        'QUANTITY': 0,
        'UNIT': 1,
        'FOOD': 2,
    }
    
    for item in data:
        # Convertir 'ingredients_entities' de cadena a lista de diccionarios
        entities = json.loads(item['ingredients_entities'])
        
        # Validar y ajustar cada entidad
        fixed_entities = []
        for entity in entities:
            if 'start' in entity and 'end' in entity and 'type' in entity:
                # Filtrar solo los tipos esperados
                if entity['type'] in type_to_index:
                    entity['start'] = int(entity['start'])
                    entity['end'] = int(entity['end'])
                    # Convertir 'type' a un índice numérico
                    entity['type'] = type_to_index[entity['type']]
                    fixed_entities.append(entity)
                else:
                    print(f"Discarding unexpected type: {entity['type']}")

        # Guardar las entidades corregidas de nuevo en el item
        item['ingredients_entities'] = fixed_entities

    return data

# Lista de archivos de conjuntos de datos para corregir
datasets = ['train', 'valid', 'test']

for dataset in datasets:
    # Cargar el conjunto de datos
    with open(f'dataset/{dataset}.json', 'r') as file:
        data = json.load(file)

    # Corregir las entidades
    data = fix_entities(data)

    # Guardar el conjunto de datos corregido en un archivo JSON
    with open(f'dataset/{dataset}_fixed.json', 'w') as file:
        json.dump(data, file, indent=2)

print("Los conjuntos de datos han sido corregidos y guardados.")