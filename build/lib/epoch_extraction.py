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

        s_freq = raw.info["sfreq"] #extracts the sampling frequency

        samples_per_epoch = int(epoch_length * s_freq) #convert epoch length to samples -this calculates the number of samples in one epoch

        step_size = int (samples_per_epoch * (1-overlap)) #how far to move forward for the next epoch

        # Extract data

        data = raw.get_data()
        n_channels, n_samples = data.shape

        # Create overlapping epochs

        epochs =  [] #initiate an empty list to store the extracted epochs
        for start in range(0, n_samples - samples_per_epoch + 1, step_size):
            epoch = data[:, start:start + samples_per_epoch] #extracts samples corresponding to the epoch length recall it is time samples(columns) we are epoching and selecting all channels(rows)
            epochs.append(epoch)

        epochs_data = np.array(epochs)

    except Exception as e:
        logger.error(f"Error extracting epochs {e}")
        epochs_data = np.array([])

    return epochs_data

def write_epoch_data(epochs_data, output_folder, file_name):
    try:
        for i, epoch in enumerate(epochs_data):
            subject_id = file_name.split('_')[0]  # Extract subject ID by splitting at the underscore and selecting the first index(e.g., sub-001)
            filename = f"{subject_id}_epoch-{i+1:03d}.npy"
            epoch_file = os.path.join(output_folder, filename)
            np.save(epoch_file, epoch)  # Save the epoch as a separate file

        logger.info(f"Saved {len(epochs_data)} epochs for {file_name}") #prints the number of saved epochs for the current file
    except Exception as e:
        logger.error(f"Error saving the epoch file {e}")


def batch_extract_epochs(data_folder, output_folder):
    """
    This function 
    """
    try:
        for file_name in os.listdir(data_folder):
            if file_name.endswith(".fif"):
                file_path = os.path.join(data_folder, file_name) #combines the folder path and the file name to get the full path
                logger.info(f"Processing {file_name}...")
                #loading the fif preprocessed data
                raw = read_raw_fif(file_path, preload=True)
                #calling the extract epoch function
                epochs_data = extract_epochs(raw, epoch_length=4, overlap = 0.5)
                # Save each epoch as a separate file
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