import pandas as pd
import os
from load_data      import load_data
from preprocessing  import PREPROCESSING_METHODS

def main():
    print("--- Step 1: Loading & Merging Datasets ---")
    df = load_data()
    
    text_column = 'text' 
    print(f"Success! Loaded {len(df)} total tweets.")
    print(df['sentiment'].value_counts()) # Show counts for each sentiment class
    print("-" * 50)

    print("--- Step 2: Running Preprocessing Pipelines ---")
    
    # Apply all 6 methods dynamically from your preprocessing.py dictionary
    for name, preprocess_function in PREPROCESSING_METHODS.items():
        print(f"Applying method: {name} ...")
        df[f'cleaned_{name}'] = df[text_column].apply(preprocess_function)
        
    print("-" * 50)
    print("--- Step 3: Saving Cleaned Data ---")
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/tweets_preprocessed.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing step completed successfully!")
    print(f"All 6 variations saved into: '{output_path}'")

if __name__ == "__main__":
    main()