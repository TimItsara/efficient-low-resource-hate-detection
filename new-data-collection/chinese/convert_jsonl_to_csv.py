import json
import csv
import pandas as pd

def jsonl_to_csv(jsonl_file, csv_file):
    """
    Convert JSONL file to CSV format.
    
    Args:
        jsonl_file: Path to input JSONL file
        csv_file: Path to output CSV file
    """
    # Read JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                obj = json.loads(line)
                # Flatten the nested meta dictionary
                flat_obj = {
                    'text': obj.get('text', ''),
                    'label': obj.get('label', ''),
                    'source': obj.get('source', '')
                }
                # Add meta fields if they exist
                if 'meta' in obj:
                    for key, value in obj['meta'].items():
                        flat_obj[f'meta_{key}'] = value
                
                data.append(flat_obj)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Successfully converted {jsonl_file} to {csv_file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    jsonl_file = "chinese_8000.jsonl"
    csv_file = "chinese_8000.csv"
    jsonl_to_csv(jsonl_file, csv_file)
