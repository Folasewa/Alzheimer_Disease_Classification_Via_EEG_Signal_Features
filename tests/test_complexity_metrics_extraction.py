from src.complexity_metrics_extraction import extract_entropy_features, process_epoch_file
from src.utilities import checkpoint
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


def test_extract_entropy_features_pos():
    epoch_file = "/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    subject_id = "sub-001"
    epoch_number = 1
    features = extract_entropy_features(epoch, subject_id, epoch_number)
    assert features is not None

def test_extract_entropy_features_neg():
    epoch_file = "/test_data/epochs_overlap/001_epo-001.npy"
    epoch = np.load(epoch_file)
    subject_id = "001"
    epoch_number = 1
    features = extract_entropy_features(epoch, subject_id, epoch_number)
    assert features is None

def test_process_epoch_file_pos():
    complexity_file_csv = "test_data/complex.csv"
    output_folder = "test_data/epochs_overlap"
    files_to_process, processed_set = checkpoint(complexity_file_csv, output_folder)

    features_df = Parallel(n_jobs=-1)(
        delayed(process_epoch_file)(epochs_file, output_folder, complexity_file_csv, processed_set)
        for epochs_file in files_to_process)
    assert len(features_df) > 0

def test_process_epoch_file_neg():
    complexity_file_csv = "test_data/comp.csv"
    output_folder = "test_data/epochs_overlap"
    files_to_process, processed_set = checkpoint(complexity_file_csv, output_folder)

    features_df = Parallel(n_jobs=-1)(
        delayed(process_epoch_file)(epochs_file, output_folder, complexity_file_csv, processed_set)
        for epochs_file in files_to_process)
    assert len(features_df) == 0
