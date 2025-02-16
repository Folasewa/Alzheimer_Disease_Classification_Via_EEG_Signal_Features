from logger import setup_logger
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from EntropyHub import ApEn, SampEn, PermEn
from utilities import checkpoint

logger = setup_logger("logger", "logs.log")

def extract_entropy_features(epoch, subject_id, epoch_number):

    """
    Compute entropy features: Approximate Entropy, Sample Entropy,
    Permutation Entropy, and Multiscale Entropy for one epoch.

    Parameters:
    - epoch: 2D array (n_channels, n_samples) for the epoch
    - subject_id: ID of the subject
    - epoch_number: Epoch index number
    Returns:
    - features: Dictionary with entropy values for each channel and metadata
    """
    try:
        n_channels = epoch.shape[0]
        features = {
            "subject_id": subject_id,
            "epoch_number": epoch_number
        }

        for ch_idx in range(n_channels):
            channel_data = epoch[ch_idx]

            # Calculate Approximate Entropy (ApEn)
            ap_en = ApEn(channel_data, m=2, r=0.15 * np.std(channel_data))[0]
            features[f"channel_{ch_idx}_ApEn"] = ap_en

            # Calculate Sample Entropy (SampEn)
            samp_en = SampEn(channel_data, m=2, r=0.15 * np.std(channel_data))[0]
            features[f"channel_{ch_idx}_SampEn"] = samp_en

            # Calculate Permutation Entropy (PermEn)
            perm_en = PermEn(channel_data, m=2, tau=5)[0]
            features[f"channel_{ch_idx}_PermEn"] = perm_en

    except Exception as e:
        logger.error(f"Error extracting entropy features for channel: {e}")
        features = {}
    return features

def process_epoch_file(epochs_file, output_folder, complexity_file_csv, processed_set):

    """
    Process a single epoch file to extract entropy features and append them to the CSV file.

    Parameters:
    - epochs_file: Path to the .npy epoch file
    - output_folder: Path to the folder containing epochs
    - complexity_file_csv: Path to the output CSV file
    - processed_set: Set of already processed (subject_id, epoch_number) tuples
    """

    try:
        epoch_path = os.path.join(output_folder, epochs_file)
        epoch_data = np.load(epoch_path)

        # Extract subject ID and epoch number
        subject_id = epochs_file.split('_')[0]
        epoch_number = int(epochs_file.split('_')[1].replace('epoch-', '').split('.')[0])
        # Skip if the file has already been processed
        if (subject_id, epoch_number) in processed_set:
            print(f"Epoch {epoch_number} for subject {subject_id} already processed. Skipping...")
            return
        # Extract features
        features = extract_entropy_features(epoch_data, subject_id, epoch_number)
        features_df = pd.DataFrame([features])
        if not os.path.isfile(complexity_file_csv) or os.path.getsize(complexity_file_csv) == 0:
            # Write with headers if file doesn't exist or is empty
            features_df.to_csv(complexity_file_csv, mode='w', index=False, header=True)
        else:
            # Append without headers if the file already exists and is not empty
            features_df.to_csv(complexity_file_csv, mode='a', index=False, header=False)
        logger.info(f"Processed and saved: {epochs_file}")

    except Exception as e:
        logger.error(f"Error processing and extracting entropy features for: {e}")
        features_df = pd.DataFrame([])

    return features_df

def main():
    complexity_file_csv = "data/complex.csv"
    output_folder = "data/epochs_overlap"
    files_to_process, processed_set = checkpoint(complexity_file_csv, output_folder)
    features_df = Parallel(n_jobs=-1)(
        delayed(process_epoch_file)(epochs_file, output_folder, complexity_file_csv, processed_set)
        for epochs_file in files_to_process)
    logger.info(f"All entropy features saved to {complexity_file_csv}")

if __name__ == "__main__":
    main()
