import os
import pandas as pd

def checkpoint(csv_file, output_folder):
    """
    Checkpoint the processed data to disk.
    """
    # Precompute processed files upfront
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        existing_df = pd.read_csv(csv_file)[['subject_id', 'epoch_number']]
        processed_set = set(
            (row['subject_id'], row['epoch_number']) for _, row in existing_df.iterrows())
    else:
        processed_set = set()

    # List files to process
    files_to_process = [
        f for f in os.listdir(output_folder) if f.endswith('.npy')
    ]
    return files_to_process, processed_set
