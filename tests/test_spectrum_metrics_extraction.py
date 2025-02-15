from src.spectrum_metrics_extraction import compute_basic_statistics, compute_psd, compute_relative_band_power, extract_spectrum_features, batch_extract_spectrum_features
import numpy as np
import pandas as pd


def test_compute_basic_statistics_pos():

    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    stats_features = compute_basic_statistics(epoch)
    assert not pd.isnull(stats_features)

def test_compute_basic_statistics_neg():

    epoch_file = "tests/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.array([])
    stats_features = compute_basic_statistics(epoch)
    assert not stats_features

def test_compute_psd_pos():
    sfreq = 500
    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    psd_features = compute_psd(epoch, sfreq)
    assert not pd.isnull(psd_features)

def test_compute_psd_neg():
    sfreq = 500
    epoch_file = "tests/sub-001_epoch-001.npy"
    epoch = np.array([])
    psd_features = compute_psd(epoch, sfreq)
    assert not psd_features

def test_compute_relative_band_power_pos():
    sfreq = 500
    n_channels = 19
    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    psd_features = compute_psd(epoch, sfreq)
    relative_band_power = compute_relative_band_power(psd_features, n_channels)
    assert not pd.isnull(relative_band_power)

def test_compute_relative_band_power_neg():
    sfreq = 500
    n_channels = 19
    epoch_file = "tests/test_data/non-existent-path/sub-001_epoch-001.npy"
    epoch = np.array([])
    psd_features = compute_psd(epoch, sfreq)
    relative_band_power = compute_relative_band_power(psd_features, n_channels)
    assert not relative_band_power

def test_extract_spectrum_features_pos():
    sfreq = 500
    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    features = extract_spectrum_features(epoch, sfreq)
    assert features

def test_extract_spectrum_features_neg():
    sfreq = 500
    epoch_file = "tests/wrong-folder/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.array([])
    features = extract_spectrum_features(epoch, sfreq)
    assert not features

def test_batch_extract_spectrum_features_pos():
    output_folder = "tests/test_data/epochs_overlap"
    spectrum_file_csv = "tests/test_data/spectrum.csv"
    sfreq = 500
    spectrum_features_df = batch_extract_spectrum_features(output_folder, spectrum_file_csv, sfreq)
    assert not spectrum_features_df.empty

def test_batch_extract_spectrum_features_neg():
    output_folder = "tests/data_test/epochs_overlap"
    spectrum_file_csv = "tests/wrong_folder/spectrum.csv"
    sfreq = 500
    spectrum_features_df = batch_extract_spectrum_features(output_folder, spectrum_file_csv, sfreq)
    assert spectrum_features_df.empty
