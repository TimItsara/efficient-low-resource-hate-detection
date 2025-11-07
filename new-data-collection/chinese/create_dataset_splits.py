import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

def create_dataset_structure(input_csv, output_dir, train_sizes=[10, 20, 30, 40, 100, 200, 300, 1000, 2000], 
                            random_seeds=range(1, 11), dev_size=500, test_size=2000):
    """
    Convert CSV to dataset structure similar to thai_hatesent example.
    
    Args:
        input_csv: Path to input CSV file (ChineseHate.csv)
        output_dir: Output directory name (e.g., 'chinese_hatesent')
        train_sizes: List of training set sizes
        random_seeds: Range of random seeds for multiple splits
        dev_size: Size of development set
        test_size: Size of test set
    """
    # Read the data
    df = pd.read_csv(input_csv)
    
    # Keep only text and label columns
    df_simple = df[['text', 'label']].copy()
    
    # Create output directory structure
    base_dir = output_dir
    train_dir = os.path.join(base_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    print(f"Creating dataset structure in: {base_dir}")
    print(f"Total samples: {len(df_simple)}")
    
    # First, split off test set (fixed)
    remaining_data, test_data = train_test_split(
        df_simple, 
        test_size=test_size, 
        random_state=42,
        stratify=None  # Can stratify by label if needed
    )
    
    # Then split remaining into train and dev
    train_full, dev_data = train_test_split(
        remaining_data,
        test_size=dev_size,
        random_state=42
    )
    
    # Save dev and test sets (fixed)
    dev_data.to_csv(os.path.join(base_dir, f'dev_{dev_size}.csv'), index=False)
    test_data.to_csv(os.path.join(base_dir, f'test_{test_size}.csv'), index=False)
    
    # Save full training set
    train_full.to_csv(os.path.join(base_dir, f'train_{len(train_full)}.csv'), index=False)
    
    print(f"✓ Created dev_{dev_size}.csv ({len(dev_data)} samples)")
    print(f"✓ Created test_{test_size}.csv ({len(test_data)} samples)")
    print(f"✓ Created train_{len(train_full)}.csv ({len(train_full)} samples)")
    
    # Create different sized training sets with multiple random seeds
    print("\nCreating training subsets with different sizes and random seeds...")
    
    for size in train_sizes:
        if size > len(train_full):
            print(f"⚠ Skipping train_{size} (larger than available training data)")
            continue
            
        for seed in random_seeds:
            # Sample from full training set
            train_subset = train_full.sample(n=size, random_state=seed)
            
            # Save to train directory
            filename = f'train_{size}_rs{seed}.csv'
            train_subset.to_csv(os.path.join(train_dir, filename), index=False)
        
        print(f"✓ Created {len(random_seeds)} versions of train_{size} (rs1-rs{max(random_seeds)})")
    
    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    print(f"\nStructure:")
    print(f"  {base_dir}/")
    print(f"    ├── dev_{dev_size}.csv")
    print(f"    ├── test_{test_size}.csv")
    print(f"    ├── train_{len(train_full)}.csv")
    print(f"    └── train/")
    for size in train_sizes:
        if size <= len(train_full):
            print(f"        ├── train_{size}_rs1.csv to train_{size}_rs{max(random_seeds)}.csv")

if __name__ == "__main__":
    # Configuration
    input_csv = "chinese_8000.csv"
    output_dir = "chinese_hatesent"
    
    # Create the dataset structure
    create_dataset_structure(
        input_csv=input_csv,
        output_dir=output_dir,
        train_sizes=[10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000],
        random_seeds=range(1, 11),  # 10 different random seeds
        dev_size=500,
        test_size=2000
    )
