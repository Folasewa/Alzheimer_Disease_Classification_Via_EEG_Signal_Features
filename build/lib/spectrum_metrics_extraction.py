from logger import setup_logger
import pandas as pd
import numpy as np
import os
from scipy.signal import welch
logger = setup_logger("logger", "logs.log")
# FIRST STEP:

# Importing libraries
def compute_basic_statistics(epoch):

    """
    Compute the basic statistics (mean, variance and interquartile range)for each channel in an epoch
    paramters:
    epoch: 2D array of shape (n_channels, n_samples) for one epoch
    returns:
    -stats_features: Dictionary with mean, variance and IQR values for each channel
    """
    try:
        n_channels = epoch.shape[0]
        stats_features = {}
        for ch_idx in range(n_channels):
            #calculate mean
            mean_value = np.mean(epoch[ch_idx])
            stats_features[f"channel_{ch_idx}_mean"] = mean_value

            #calculate variance
            variance_value = np.var(epoch[ch_idx])
            stats_features[f"channel_{ch_idx}_variance"] = variance_value

            #calculate IQR
            IQR_value = np.percentile(epoch[ch_idx], 75) - np.percentile(epoch[ch_idx], 25)
            stats_features[f"channel_{ch_idx}_IQR"] = IQR_value
    except Exception as e:
            logger.error(f"Error computing stats {ch_idx}: {e}")
            stats_features = {}
    return stats_features

def compute_psd(epoch, sfreq):

    """
    Compute the Power Spectral Density (PSD) for the full frequency range (0.5-45 hz)
    Compute the Power Spectral Density (PSD) for each EEG rhythm in the frequency band

    Parameters:
    -epoch: 2D array of shape (n_channels, n_samples) for one epoch
    -sfreq; sampling frequency

    Returns:
    -psd_features: Dictionary with PSD values for each frequency band and channel

    """
    try:

        n_channels, n_samples = epoch.shape
        psd_features = {}
        # Define the bands as a dictionary, {band: (fmin, fmax)}
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 25),
            'gamma': (25, 45),
        }

        for ch_idx in range(n_channels):
            freqs, psd = welch(epoch[ch_idx], fs=sfreq, nperseg=n_samples)

            # Extract PSD for the full frequency range
            psd_full_range = np.sum(psd[(freqs >=0.5) & (freqs <=45)]) #gets the total power within the full range
            psd_features[f"channel_{ch_idx}_psd_full"] = psd_full_range

            # Extract PSD for each EEG rhythm (delta, theta, etc)

            for band, (fmin, fmax) in bands.items(): #iterating over the key and value
                band_power = np.sum(psd[(freqs >= fmin) & (freqs < fmax)]) #calculates total power for each band
                psd_features[f"channel_{ch_idx}_psd_{band}"] = band_power
    except Exception as e:
        logger.error(f"Error computing PSD {e}")
        psd_features = {}

    return psd_features

# Step 3: Relative Band Power: This step normalizes the power in each frequency band relative to the total power

def compute_relative_band_power(psd_features, n_channels):

    """
    Compute relative band power in each channel

    Parameters:

    -psd_features: a dictionary of PSD values for each band and channel
    -n_channels: number of EEG channels

    Returns:

    -relative band power: dictionary with relative power for each band and channel

    """
    try:

        relative_band_power = {}

        for ch_idx in range (n_channels):
                total_power = psd_features[f"channel_{ch_idx}_psd_full"]
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    band_power = psd_features[f"channel_{ch_idx}_psd_{band}"]
                    relative_band_power[f"channel_{ch_idx}_relative_{band}"] = band_power / total_power
    except Exception as e:
            logger.error(f"Error computing relative band power for channel {ch_idx}: {e}")
            relative_band_power = {}
    return relative_band_power

# Step4: Combine all these metrics in one function to extract spectrum features

def extract_spectrum_features(epoch, sfreq):

    """
    Extract spectrum features: mean, variance, IQR, PSD and relative band power

    parameters

    epoch: 2D array of shape
    sfreq: sampling frequency

    returns:

    -features: Dictionary of spectrum features for each channel

    """
    try:

        n_channels = epoch.shape[0]
        features = {}
            # Compute the time-domain metrics (Basic statistics)
        features.update(compute_basic_statistics(epoch))
            # Compute psd
        psd_features = compute_psd(epoch, sfreq)
        features.update(psd_features)
      # Compute relative band power
        relative_band_power = compute_relative_band_power(psd_features, n_channels)
        features.update(relative_band_power)
    except Exception as e:
        features = {}
        logger.error(f"Error extracting spectrum features {e}")

    return features

def batch_extract_spectrum_features(output_folder, spectrum_file_csv, sfreq):
    """
    Batch extracts spectrum features from all epochs
    """
    try:
        spectrum_features = []
        # Looping through the epochs folder
        for epochs_file in os.listdir(output_folder):
            if epochs_file.endswith('.npy'):
                    epoch_path = os.path.join(output_folder, epochs_file)
                    epoch_data = np.load(epoch_path)
                    # Extract the spectrum features
                    features = extract_spectrum_features(epoch_data, sfreq)
                    # Add metadata (subject ID and epoch number)
                    subject_id = epochs_file.split('_')[0]  # Extract subject ID (e.g., "sub-001")
                    epoch_number = epochs_file.split('_')[1].replace('epoch-', '').split('.')[0]
                    features['subject_id'] = subject_id
                    features['epoch_number'] = int(epoch_number)
                    # Append the features to the list
                    spectrum_features.append(features)
        # Convert the list to dataframe
        spectrum_features_df = pd.DataFrame(spectrum_features)
        # Save to the csv file
        spectrum_features_df.to_csv(spectrum_file_csv, index=False)
        logger.info(f"Spectrum features saved to {spectrum_file_csv}")
    except Exception as e:
        spectrum_features_df = pd.DataFrame([])
        logger.error(f"Error extracting features {e}")
    return spectrum_features_df

def main():
    spectrum_file_csv = "data/spectrum.csv"
    output_folder = "data/epochs_overlap"
    spectrum_features = []
    sfreq = 500
    spectrum_features_df = batch_extract_spectrum_features(output_folder, spectrum_file_csv, sfreq)
    logger.info(spectrum_features_df.head(5))

if __name__ == "__main__":
    main()