# Meal-to-Ingredients Model

Este proyecto tiene como objetivo desarrollar un modelo de procesamiento de lenguaje natural (NLP) para identificar y clasificar entidades en textos relacionados con recetas, como cantidades, unidades y tipos de alimentos, utilizando `DistilBERT` para la clasificación de tokens.

## Estructura del Proyecto

- `.dvc/` - Archivos de configuración de `DVC`.
- `.gitignore` - Archivo de configuración para ignorar archivos en el control de versiones.
- `dataset/` - Contiene los datasets transformados y listos para entrenamiento y evaluación.
- `model_output/` - Directorio donde se guardó el modelo entrenado y el tokenizador.
- `results/` - Directorio donde se guardaron los resultados de las evaluaciones.
- `evaluate_model.py` - Script para evaluar el modelo con datos de prueba.
- `fix-datasets.py` - Script para transformar los datasets a un formato más adecuado.
- `save-tokenizer.py` - Script para guardar el tokenizador asociado al modelo.
- `simplification.py` - Script para simplificar la estructura de los datos.
- `test_model.py` - Script para realizar pruebas rápidas con ejemplos específicos.
- `train_model.py` - Script para entrenar el modelo.

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone git@github.com:evilvic/meal-to-ingredients-MODEL.git
   cd meal-to-ingredients-MODEL
   ```

2. **Crear y activar el entorno virtual**:
   ```bash
   python3 -m venv meal_to_ingredients_env
   source meal_to_ingredients_env/bin/activate
   ```

3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Instalar `DVC`**:
   ```bash
   pip install dvc
   ```

## Preparación del Dataset

El dataset fue transformado para que cada muestra contenga una oración con los ingredientes y las etiquetas correspondientes para cada palabra o token en la oración.

Ejemplo del formato final:
```json
[
  {
    "sentence": "5 ounces rum",
    "labels": [
      {"word": "5", "label": "B-QUANTITY"},
      {"word": "ounces", "label": "B-UNIT"},
      {"word": "rum", "label": "B-FOOD"}
    ]
  }
]
```

## Entrenamiento del Modelo

Para entrenar el modelo, usa el script `train_model.py`:

```bash
python train_model.py
```

Esto entrenará el modelo `DistilBERT` y lo guardará en el directorio `model_output/`.

## Evaluación del Modelo

Puedes evaluar el modelo usando el script `evaluate_model.py`:

```bash
python evaluate_model.py
```

Esto generará un informe de clasificación que muestra la precisión, el recall y la puntuación F1 para cada clase (QUANTITY, UNIT, FOOD).

### Resultados de la Evaluación

En la última evaluación, el modelo presentó los siguientes resultados:

```
precision    recall  f1-score   support

    FOOD       0.00      0.00      0.00       101
QUANTITY       0.27      1.00      0.43        66
    UNIT       0.00      0.00      0.00        76

micro avg      0.27      0.27      0.27       243
macro avg      0.09      0.33      0.14       243
weighted avg   0.07      0.27      0.12       243
```

## Pruebas con Nuevos Ejemplos

Puedes probar el modelo con nuevas oraciones usando el script `test_model.py`:

```bash
python test_model.py
```

Este script toma una oración de prueba y muestra los tokens y sus etiquetas predichas.

## Control de Versiones con DVC

`DVC` se usa para rastrear cambios en los datasets y en los resultados del modelo. Puedes agregar datasets y resultados bajo control de `DVC` utilizando los siguientes comandos:

```bash
dvc add dataset/train_transformed.json
dvc add dataset/valid_transformed.json
dvc add dataset/test_transformed.json
dvc add model_output/
dvc add results/
```

Luego, puedes hacer commit de estos cambios con `Git`:

```bash
git add .
git commit -m "Add transformed datasets and model output"
```

## Próximos Pasos

1. **Mejora del Modelo**: Se sugiere mejorar la arquitectura del modelo o realizar técnicas de aumento de datos para mejorar el rendimiento en las clases `FOOD` y `UNIT`.
2. **Ajuste de Hiperparámetros**: Realizar más experimentos para ajustar los hiperparámetros y mejorar la precisión general del modelo.
3. **Experimentación con DVC**: Usar `DVC` para gestionar y comparar diferentes experimentos.

## Contacto

Para más información o contribuciones, por favor contacta al mantenedor del proyecto.
