import logging
from logger import setup_logger
import numpy as np
import os
from mne.io import read_raw_fif
logger = setup_logger("logger", "logs.log")


def extract_epochs(raw, epoch_length, overlap):

    """
    Extract 4-second epochs with 50% overlap from the eeg data
    parameters:
    -raw: mne.io
    -epoch_length: length of each epoch in seconds, in this case it is 4 seconds
    -overlap: 50%

    returns:
    -epochs_data: array of shape (n_epochs, n_channels, n_samples)

    """
    try:
        s_freq = raw.info["sfreq"]
        samples_per_epoch = int(epoch_length * s_freq)
        step_size = int (samples_per_epoch * (1-overlap))
        data = raw.get_data()
        n_channels, n_samples = data.shape
        # Create overlapping epochs
        epochs =  []
        for start in range(0, n_samples - samples_per_epoch + 1, step_size):
            epoch = data[:, start:start + samples_per_epoch]
            epochs.append(epoch)
        epochs_data = np.array(epochs)
    except Exception as e:
        logger.error(f"Error extracting epochs {e}")
        epochs_data = np.array([])
    return epochs_data

def write_epoch_data(epochs_data, output_folder, file_name):
    """
    Saves extracted epochs
    """
    try:
        for i, epoch in enumerate(epochs_data):
            subject_id = file_name.split('_')[0]
            filename = f"{subject_id}_epoch-{i+1:03d}.npy"
            epoch_file = os.path.join(output_folder, filename)
            np.save(epoch_file, epoch)
        logger.info(f"Saved {len(epochs_data)} epochs for {file_name}")
    except Exception as e:
        logger.error(f"Error saving the epoch file {e}")

def batch_extract_epochs(data_folder, output_folder):
    """
    This function batch extract epochs from a folder
    """
    try:
        for file_name in os.listdir(data_folder):
            if file_name.endswith(".fif"):
                file_path = os.path.join(data_folder, file_name)
                logger.info(f"Processing {file_name}...")
                raw = read_raw_fif(file_path, preload=True)
                epochs_data = extract_epochs(raw, epoch_length=4, overlap = 0.5)
                write_epoch_data(epochs_data, output_folder, file_name)
    except Exception as e:
        logger.error(f"Error extracting Epoch {e}")
        epochs_data = np.array([])
    return epochs_data

def main():
    data_folder = "data/preprocessed_filtered"
    output_folder = "data/epochs_overlap"

    # Main processing
    epochs_data = batch_extract_epochs(data_folder, output_folder)

if __name__ == "__main__":
    main()