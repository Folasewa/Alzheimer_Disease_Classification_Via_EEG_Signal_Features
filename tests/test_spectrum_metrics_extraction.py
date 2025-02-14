from src.spectrum_metrics_extraction import compute_basic_statistics, compute_psd, compute_relative_band_power, extract_spectrum_features, batch_extract_spectrum_features
import numpy as np


def test_compute_basic_statistics():

    epoch_file = "/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    stats_features = compute_basic_statistics(epoch)
    assert stats_features is not None

def test_compute_psd():
    sfreq = 500
    epoch_file = "/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    psd_features = compute_psd(epoch, sfreq)
    assert psd_features is not None

def test_compute_relative_band_power():
    sfreq = 500
    n_channels = 19
    epoch_file = "/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    psd_features = compute_psd(epoch, sfreq)
    relative_band_power = compute_relative_band_power(psd_features, n_channels)
    assert relative_band_power is not None

def test_extract_spectrum_features():
    sfreq = 500
    epoch_file = "/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    features = extract_spectrum_features(epoch, sfreq)
    assert features is not None

def test_batch_extract_spectrum_features():
    output_folder = "test_data/epochs_overlap"
    spectrum_file_csv = "test_data/spectrum.csv"
    sfreq = 500
    spectrum_features_df = batch_extract_spectrum_features(output_folder, spectrum_file_csv, sfreq)
    assert spectrum_features_df is not None