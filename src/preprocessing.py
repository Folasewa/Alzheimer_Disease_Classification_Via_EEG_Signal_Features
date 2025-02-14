import os
import logging
import glob
import numpy as np
import pandas as pd
import shutil
from scipy.signal import welch
from mne.io import read_raw_eeglab
from mne.preprocessing import annotate_amplitude, ICA
from joblib import Parallel, delayed


def read_data(data_path):
    """
    Read EEG data from a file.
    """
    try:
        data = read_raw_eeglab(data_path, preload=True)
    except Exception as e:
        data = None
        logging.error(f"Error reading data: {e}")
    return data


def copy_data(source_path, destination_path, limit=65, zfill=3):
    """
    Copy EEG data from source to destination folder.
    """
    try:
        for subject_id in range(1, limit):
            subject_folder = f"sub-{str(subject_id).zfill(zfill)}"  # the default zfill is 3, because the format is in sub-001            
            src_pth = f"{source_path}/{subject_folder}/eeg/{subject_folder}_task-eyesclosed_eeg.set"

            if os.path.exists(src_pth):
                # print("+++++++++++++++++++++", source_path, "*********************", destination_path)
                shutil.copy(src_pth, destination_path)
            else:
                logging.info(f"File not found {src_pth}")
    except Exception as e:
        logging.error(f"Error copying data: {e}")


def check_noise(dataset, noise_threshold, n_components):
    """
    Check noise in EEG data.
    """
    try:
        # Standard deviation is computed for each channel and its max is taken for comparison
        channel_stds = dataset.get_data().std(axis=1)
        max_std = channel_stds.max()

        # If noise detected, apply ICA
        if max_std > noise_threshold:
            logging.info(f"Noise detected! Maximum channel std: {max_std:.2e} V. Applying ICA...")

            # Perform ICA to remove artifacts
            ica = ICA(n_components=n_components, random_state=42, max_iter="auto")
            ica.fit(dataset)
            # This automatically excludes the artifacts if found
            ica.exclude = []
            dataset = ica.apply(dataset)
        else:
            logging.info(f"No significant noise detected (max std: {max_std:.2e} V). Skipping ICA.")
    except Exception as e:
        logging.error(f"Error checking noise: {e}")
        dataset = None
    return dataset


def exclude_bad_segments(dataset):
    """
    Exclude bad segments from EEG data.
    """
    try:
        good_data = dataset.get_data()
        for annotation in dataset.annotations:
            onset_sample = int(annotation["onset"] * dataset.info["sfreq"])  # this gets the start of the bad segment
            duration_sample = int(annotation["duration"] * dataset.info["sfreq"])  # this gets the length of the bad segment
            good_data[:, onset_sample : onset_sample + duration_sample] = (
                np.nan
            )  # slices the data for the time range and marks bad segments as NaN
    except Exception as e:
        logging.error(f"Error exluding bad segment: {e}")
        good_data = None
    return good_data


def write_data(data, source_path, output_path):
    try:
        # Save the preprocessed file
        output_file = os.path.join(output_path, os.path.basename(source_path).replace(".set", "_preprocessed.fif"))
        logging.info(f"Saving the preprocessed file to: {output_file}")
        data.save(output_file, overwrite=True)
        written = True
    except Exception as e:
        logging.error(f"Error writing data: {e}")
        written = False
    return written

def preprocess_file(
    source_path,
    output_path,
    n_components=19,
    noise_threshold=6e-5,
    l_freq=0.5,
    h_freq=45,
    artifact_peak=17e-6,
    artifact_min_duration=0.5,
):
    """
    Preprocess an EEG file with optional noise detection, ICA, and ASR.

    Parameters:
    - source_path: Path to the input .set file.
    - output_path: Path to save the preprocessed file.
    - n_components: Number of components for ICA.
    - noise_threshold: Noise threshold in volts (default is 30 µV, converted to volts).
    """
    try:
        logging.info(f"Processing file: {source_path}")

        # This loads the raw .set file since eeglab format for eeg is .set
        raw_dataset = read_raw_eeglab(source_path, preload=True)

        # The reference electrodes A1 and A2 were absent, hence average re-referencing was done
        logging.info("Applying average re-referencing...")
        raw_dataset.set_eeg_reference(ref_channels="average")

        # A band-pass filter of 0.5 to 45Hz was applied (as per the literature)
        logging.info("Applying band-pass filter 0.5Hz to 45Hz...")
        raw_dataset.filter(l_freq=l_freq, h_freq=h_freq)

        # This checks for noise in the signal
        logging.info("Checking for noise in the data...")

        raw_dataset = check_noise(raw_dataset, noise_threshold, n_components)

        # Apply ASR for automatic artifact rejection
        logging.info("Applying Artifact Subspace Reconstruction (ASR)...")

        # Annotate bad segments based on the amplitude threshold
        annotations, _ = annotate_amplitude(
            raw_dataset,
            peak=artifact_peak,  # threshold for detecting artifacts
            min_duration=artifact_min_duration,  # Minimum artifact duration (0.5 seconds)
        )
        raw_dataset.set_annotations(annotations)  # annotations added to the dataset to mark bad data for exclusion

        # Exclude bad segments manually
        logging.info("Excluding bad segments...")
        good_data = exclude_bad_segments(raw_dataset)

        # Creating a cleaned copy of the data
        raw_dataset_clean = raw_dataset.copy()
        raw_dataset_clean._data = np.nan_to_num(good_data, nan=0.0)  # Replacing the Nans with zeroes

        write_data(raw_dataset_clean, source_path, output_path)

    except Exception as e:
        logging.error(f"Unexpected error in processing file {e}")
        raw_dataset_clean = None

    return raw_dataset, raw_dataset_clean

def main():
    # Define paths
    raw_dataset_folder = ("data/ds004504")
    subject_path = "data/filtered_subjects"
    source_path = glob.glob(f"{subject_path}/*.set")
    output_path = "data/preprocessed_filtered"

    # copy_data(raw_dataset_folder, subject_path, 66, 3)
    # Preprocess EEG data in parallel
    Parallel(n_jobs=-1)(
        delayed(preprocess_file)(file, output_path) for file in source_path
    )

if __name__ == "__main__":
    main()
