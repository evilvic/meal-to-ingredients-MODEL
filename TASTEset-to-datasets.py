import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset/TASTEset.csv')

# Shuffle and split the dataset
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Optionally, limit the number of samples for quicker testing
train_df = train_df.sample(60, random_state=42)
valid_df = valid_df.sample(20, random_state=42)
test_df = test_df.sample(20, random_state=42)

# Convert to dictionary and ensure clean JSON output
def save_clean_json(df, filename):
    records = df.to_dict(orient='records')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

# Save the datasets
save_clean_json(train_df, 'dataset/train.json')
save_clean_json(valid_df, 'dataset/valid.json')
save_clean_json(test_df, 'dataset/test.json')