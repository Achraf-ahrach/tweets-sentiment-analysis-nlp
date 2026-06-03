import pandas as pd
import os

def load_data():
    # Define files and their corresponding sentiment labels
    files_mapping = {
        'processedNegative.csv': 'negative',
        'processedNeutral.csv': 'neutral',
        'processedPositive.csv': 'positive'
    }
    
    dfs = []
    
    # Check both common folder names just to be completely safe
    for folder in ['p00_tweets', 'data']:
        if os.path.exists(folder):
            data_folder = folder
            break
        else:
            raise FileNotFoundError("Could not find either 'p00_tweets' or 'data' folders containing the CSV files.")

    print(f"Reading dataset files from the '{data_folder}' directory...")

    for filename, label in files_mapping.items():
        file_path = os.path.join(data_folder, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")
            
        # 1. Read the horizontal row file without headers
        df = pd.read_csv(file_path, header=None)
        
        # 2. Transpose (.T) it to make it a normal vertical column
        df = df.T
        
        # 3. Rename column to 'text' and add the 'sentiment' label column
        df.columns = ['text']
        df['sentiment'] = label
        
        dfs.append(df)
        
    # Combine negative, neutral, and positive datasets into one single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df