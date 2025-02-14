import os
import glob
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from preprocessing import copy_data, preprocess_file
from epoch_extraction import batch_extract_epochs
from spectrum_metrics_extraction import batch_extract_spectrum_features
from complexity_metrics_extraction import process_epoch_file
from complexity_preprocessing_optimization import preprocess_entropy_metrics
from synchronization_metrics_extraction import process_and_save_sync_epoch
from classification_model import classification_pipeline as train_and_evaluate_models
from plot_viz import plot_frequency_domain, plot_time_frequency_domain, load_eeg_data, plot_correlation_matrix
from statistical_test import compute_t_test_across
from logger import setup_logger

logger = setup_logger("logger", "logs.log")

data_dir = "data"
raw_dataset_folder = "data/ds004504"
subject_path = "data/filtered_subjects"
output_path = "data/preprocessed_filtered"
epoch_output_folder = "data/epochs_overlap"
spectrum_file_csv = "data/spectrum.csv"
complexity_file_csv = "data/complex.csv"
cleaned_complexity_file = "data/complexity_csv_file.csv"
synchronization_file_csv = "data/synchronization.csv"
sfreq = 500  # EEG sampling frequency

def main():
    """
    This function runs the entire EEG pipeline from preprocessing to classification
    """
    
    logger.info("Copying EEG data from the original dataset folder to a subset data folder")
    copy_data(raw_dataset_folder, subject_path, limit=66, zfill=3)


    logger.info("Preprocessing the EEG Data")
    source_files = glob.glob(f"{subject_path}/*.set")
    processed_data = Parallel(n_jobs=-1)(
        delayed(preprocess_file)(file, output_path) for file in source_files
    )
    logger.info("Your EEG data preprocessing is now complete!")


    logger.info("Extracting Epochs from the preprocessed EEG data")
    epochs_data = batch_extract_epochs(output_path, epoch_output_folder)
    logger.info("Epoch extraction complete!")


    logger.info("Extracting Spectrum metrics")
    spectrum_features_df = batch_extract_spectrum_features(epoch_output_folder, spectrum_file_csv, sfreq)
    logger.info("Spectrum feature extraction complete!")


    logger.info("Extracting Complexity Features")
    files_to_process = [f for f in os.listdir(epoch_output_folder) if f.endswith('.npy')]
    processed_set = set()  # Assume an empty set for processed files

    features_df = Parallel(n_jobs=-1)(
        delayed(process_epoch_file)(epochs_file, epoch_output_folder, complexity_file_csv, processed_set)
        for epochs_file in files_to_process
    )
    logger.info("Complexity feature extraction complete!")


    logger.info("Cleaning complexity features")
    df_complexity = pd.read_csv(complexity_file_csv)
    df_cleaned = preprocess_entropy_metrics(df_complexity)
    df_cleaned.to_csv(cleaned_complexity_file, index=False)
    logger.info(f"Cleaned complexity metrics saved to {cleaned_complexity_file}")


    logger.info("Extracting synchronization metrics")
    sync_files_to_process = [f for f in os.listdir(epoch_output_folder) if f.endswith(".npy")]
    sync_method = "pearson"  # Synchronization method: "pearson" or "plv"
    threshold_value = 0.6  # 60% strongest connections

    sync_features_df = Parallel(n_jobs=-1)(
        delayed(process_and_save_sync_epoch)(file, synchronization_file_csv, sync_method, threshold_value)
        for file in sync_files_to_process
    )
    logger.info("Synchronization feature extraction complete!")


    logger.info("Feature extraction done! Moving to model training and evaluation")
    spectrum_raw_data = pd.read_csv(spectrum_file_csv)
    complexity_data = pd.read_csv(cleaned_complexity_file)
    sync_data = pd.read_csv(synchronization_file_csv)

    train_and_evaluate_models(spectrum_raw_data, complexity_data, sync_data)

    
    logger.info("Model training and evaluation complete! Plotting visualizations")
    folder_path = "data/preprocessed_filtered"
    raw_data_labelled = pd.read_csv("data/raw_labelled_data.csv")
    plot_frequency_domain(raw_data_labelled)
    plot_time_frequency_domain(raw_data_labelled)
    
    df_eeg = load_eeg_data(folder_path)
    plot_correlation_matrix(df_eeg)


    logger.info("Visualizations complete! Running statistical tests")
    compute_t_test_across(raw_data_labelled)

    logger.info("Statistical tests complete! Pipeline complete!")

if __name__ == "__main__":
    main()