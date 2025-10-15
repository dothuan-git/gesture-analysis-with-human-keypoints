import pandas as pd
import os
from typing import Tuple
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import *

# Configuration constants
CONFIG = {
    'VIDEO_PATH': 'assets/GX010016_1080_120fps.MP4',
    'OUTPUT_PATH': '',
    'FILE_A': 'data/GX010016_1080_120fps/hands_1.csv',  # Path to csv extracted from mediapipe legacy model (1.visualization)
    'FILE_B': 'data/GX010016_1080_120fps/hands_2.csv',  # Path to csv extracted from mediapipe new model (2.extract_landmarks)
}

def load_csv_files(file_a: str, file_b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load two CSV files and ensure they have the same structure and frame indices."""
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # Round confidences to 3 decimal places
    for df in (df_a, df_b):
        df['right_confidence'] = df['right_confidence'].round(1)
        df['left_confidence']  = df['left_confidence'].round(1)
    
    if df_a.columns.tolist() != df_b.columns.tolist():
        raise ValueError("CSV files must have identical column structures")
    if not df_a['frame'].equals(df_b['frame']):
        raise ValueError("CSV files must have identical frame indices")
    
    return df_a, df_b

def merge_by_highest_score(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Merge DataFrames row by row, keeping landmarks and score from the file with higher confidence per hand."""
    # Initialize empty DataFrame with same columns as input
    result = pd.DataFrame(columns=df_a.columns)
    
    # Define column groups
    right_cols = ['right_confidence'] + [col for col in df_a.columns if col.startswith('right_') and col != 'right_confidence']
    left_cols = ['left_confidence'] + [col for col in df_a.columns if col.startswith('left_') and col != 'left_confidence']
    
    # Process each row
    rows = []
    for idx in tqdm(range(len(df_a))):
        row_a = df_a.iloc[idx]
        row_b = df_b.iloc[idx]
        frame = row_a['frame']
        
        # Initialize row with frame
        new_row = {'frame': frame}
        
        # Select right hand columns based on higher confidence
        if row_a['right_confidence'] >= row_b['right_confidence']:
            for col in right_cols:
                new_row[col] = row_a[col]
        else:
            for col in right_cols:
                new_row[col] = row_b[col]
        
        # Select left hand columns based on higher confidence
        if row_a['left_confidence'] >= row_b['left_confidence']:
            for col in left_cols:
                new_row[col] = row_a[col]
        else:
            for col in left_cols:
                new_row[col] = row_b[col]
        
        rows.append(new_row)
    
    # Convert rows to DataFrame
    result = pd.DataFrame(rows, columns=df_a.columns)
    return result

def save_merged_file(df: pd.DataFrame, output_path: str) -> None:
    """Save the merged DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved merged file: {output_path}")

def main(file_a: str, file_b: str, output_path: str) -> None:
    """Merge two hand landmark CSV files based on highest confidence scores."""
    try:
        df_a, df_b = load_csv_files(file_a, file_b)
        merged_df = merge_by_highest_score(df_a, df_b)
        save_merged_file(merged_df, output_path)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    hands_1 = CONFIG['FILE_A']
    hands_2 = CONFIG['FILE_B']

    # Create output directory
    output_dir = CONFIG["OUTPUT_PATH"]
    if not os.path.exists(output_dir) or len(output_dir) == 0:
        root_path = f'workspaces/{CONFIG["VIDEO_PATH"].split("/")[-1].split(".")[0]}'
        output_dir = get_latest_or_create(root_path, 'runs')

    main(hands_1, hands_2, output_dir)
