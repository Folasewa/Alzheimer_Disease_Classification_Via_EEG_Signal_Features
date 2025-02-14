import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import seaborn as sns
from logger import setup_logger
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft


logger = setup_logger("logger", "logs.log")

def plot_group_psd(ax, smooth_freq, spline, band_edges, frequency_bands, colors, title, color):

    """
    Plots the PSD curve for a specific group with shaded frequency bands.
    """
    try:
        ax.plot(smooth_freq, spline(smooth_freq), color=color, label=title)
        
        # Shade frequency bands
        for i in range(len(frequency_bands)):
            ax.axvspan(band_edges[i], band_edges[i+1], color=colors[i], alpha=0.3)
            ax.text((band_edges[i] + band_edges[i+1]) / 2, max(spline(smooth_freq)) * 0.9,
                    frequency_bands[i], fontsize=10, ha="center")
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (µV²/Hz)")
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plots/frequency_domain_{title}.png')
    except Exception as e:
        logger.error(f"Error plotting PSD curve {e}")

def plot_frequency_domain(raw_data_labelled):
    """
    Plots the frequency-domain spectrum for CN and AD groups.
    """
    try:
        # Define frequency bands and their actual ranges
        frequency_bands = ["delta", "theta", "alpha", "beta", "gamma"]
        band_edges = [0.5, 4, 8, 13, 30, 45]  # Start of each band + last value for gamma
        colors = ["#DDA0DD", "#FFD700", "#90EE90", "#D8BFD8", "#ADD8E6"]  # Colors for shading

        # Separate CN and AD groups
        cn_group = raw_data_labelled[raw_data_labelled["label"] == 0]
        ad_group = raw_data_labelled[raw_data_labelled["label"] == 1]

        # Compute mean PSD per frequency band for CN and AD
        cn_psd_means = [cn_group[[col for col in cn_group.columns if f"psd_{band}" in col]].mean(axis=1).mean()
                        for band in frequency_bands]
        ad_psd_means = [ad_group[[col for col in ad_group.columns if f"psd_{band}" in col]].mean(axis=1).mean()
                        for band in frequency_bands]

        # Apply Gaussian smoothing to PSD values before interpolation
        cn_psd_smooth = gaussian_filter1d(cn_psd_means, sigma=1)
        ad_psd_smooth = gaussian_filter1d(ad_psd_means, sigma=1)

        # Interpolation for smooth curve
        smooth_freq = np.linspace(band_edges[0], band_edges[-1], 300)
        cn_spline = UnivariateSpline(band_edges[:-1], cn_psd_smooth, s=0.5)  
        ad_spline = UnivariateSpline(band_edges[:-1], ad_psd_smooth, s=0.5)

        # Plot CN vs AD frequency-domain spectrum
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Plot CN and AD spectra
        plot_group_psd(axes[0], smooth_freq, cn_spline, band_edges, frequency_bands, colors, 
                    "Frequency-Domain Spectrum (CN)", "blue")
        plot_group_psd(axes[1], smooth_freq, ad_spline, band_edges, frequency_bands, colors, 
                    "Frequency-Domain Spectrum (AD)", "red")

    except Exception as e:
        logger.error(f"Error plotting frequency-domain spectrum {e}")

def plot_stft(eeg_data, fs, frequency_bands, colors, title, nperseg, noverlap):
    """
    Computes and plots the Short-Time Fourier Transform (STFT) spectrogram.
    """
    try:
        # Compute STFT
        f, t, Zxx = stft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=512, window='hann')

        # Plot spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='jet')
        plt.colorbar(label="Amplitude (µV)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Time-Frequency Analysis - {title}")

        # Set axis limits
        plt.xlim(0, 8)
        plt.ylim(0, 40)

        # Add frequency band markers
        for (band, (low, high)), color in zip(frequency_bands.items(), colors):
            plt.axhline(y=low, color='white', linestyle='dotted', linewidth=1)
            plt.axhline(y=high, color='white', linestyle='dotted', linewidth=1)
            plt.text(7.5, (low + high) / 2, band, fontsize=10, color='white', verticalalignment='center')

        # plt.show()
        plt.savefig(f'plots/frequency_domain_{title}.png')
    except Exception as e:
        logger.error(f"Error plotting the spectrogram {e}")

def plot_time_frequency_domain(raw_data_labelled):
    """
    Plots the time-frequency-domain spectrum for CN and AD groups.
    """
    try:
        # Define EEG frequency bands
        frequency_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
        colors = ["#E6CCFF", "#FFE599", "#B6D7A8", "#D5A6BD", "#CFE2F3"]
        # Set EEG sampling frequency (Hz)
        fs = 500

        # Define frontal channels
        frontal_channels = ["channel_2", "channel_7", "channel_8", "channel_15", "channel_16", "channel_17", "channel_18"]

        # Extract and average EEG signals from frontal channels
        cn_group = raw_data_labelled[raw_data_labelled["label"] == 0]
        ad_group = raw_data_labelled[raw_data_labelled["label"] == 1]
        cn_eeg = cn_group[[col for col in cn_group.columns if any(ch in col for ch in frontal_channels)]].mean(axis=1).to_numpy()
        ad_eeg = ad_group[[col for col in ad_group.columns if any(ch in col for ch in frontal_channels)]].mean(axis=1).to_numpy()

        # Inputting nperseg and noverlap parameters
        nperseg = min(256, len(cn_eeg) // 4)
        noverlap = max(nperseg // 2, 1)

        # Plot STFT for CN and AD
        plot_stft(cn_eeg, fs, frequency_bands, colors, "Cognitively Normal (CN)", nperseg, noverlap)
        plot_stft(ad_eeg, fs, frequency_bands, colors, "Alzheimer's Disease (AD)", nperseg, noverlap)
    except Exception as e:
        logger.error(f"Error plotting time-frequency domain {e}")

def load_eeg_data(folder_path):
    """
    Loads EEG data, extracts relevant information, and returns a processed DataFrame.
    """
    try:
        eeg_data_list = []
        labels = []

        for filename in sorted(os.listdir(folder_path)):  # Ensure subjects are read in order
            if filename.endswith(".fif"):
                try:
                    # Extract subject ID from filename (assuming format: "sub-XXX_task-eyesclosed.fif")
                    subject_id = int(filename.split("-")[1].split("_")[0])
                except (ValueError, IndexError) as e:
                    logger.error(f"Skipping file due to unexpected name format: {filename} | Error: {e}")
                    continue  # Skip files that don't follow expected naming convention

                file_path = os.path.join(folder_path, filename)

                # Load EEG data
                raw = mne.io.read_raw_fif(file_path, preload=True)

                # Pick only EEG channels
                eeg_channels = raw.pick_types(eeg=True).ch_names
                eeg_values = raw.get_data(picks=eeg_channels)  # Shape: (n_channels, n_samples)

                # Compute mean across time for each channel (reduces data dimensionality)
                eeg_mean = np.mean(eeg_values, axis=1)  # Shape: (n_channels,)

                # Append to dataset
                eeg_data_list.append(eeg_mean)

                # Assign label (AD = 1, CN = 0)
                label = 1 if subject_id <= 36 else 0
                labels.append(label)

        # Convert to DataFrame
        df_eeg = pd.DataFrame(eeg_data_list, columns=eeg_channels)
        df_eeg["label"] = labels  # Append labels
    except Exception as e:
        logger.error(f"Error processing the EEG Dataframe {e}")
        df_eeg = pd.DataFrame([])
    
    return df_eeg  # Return processed EEG DataFrame

def plot_correlation_matrix(df_eeg):
    """
    Computes and plots correlation matrices for CN and AD EEG data.
    """

    try:

        # Separate AD and CN data
        df_AD = df_eeg[df_eeg["label"] == 1].drop(columns=["label"])
        df_CN = df_eeg[df_eeg["label"] == 0].drop(columns=["label"])

        # Compute correlation matrices
        corr_AD = df_AD.corr()
        corr_CN = df_CN.corr()

        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.tight_layout()

        # Plot for Cognitively Normal (CN)
        sns.heatmap(corr_CN, ax=axes[0], cmap="jet", annot=False, fmt=".2f")
        axes[0].set_title("Correlation Matrix - Cognitively Normal (CN)")
        axes[0].set_xlabel("Electrode")
        axes[0].set_ylabel("Electrode")
        plt.savefig('plots/frequency_domain_Cognitively Normal (CN).png')

        # Plot for Alzheimer's Disease (AD)
        sns.heatmap(corr_AD, ax=axes[1], cmap="jet", annot=False, fmt=".2f")
        axes[1].set_title("Correlation Matrix - Alzheimer's Disease (AD)")
        axes[1].set_xlabel("Electrode")
        axes[1].set_ylabel("Electrode")
        plt.savefig('plots/frequency_domain_Alzheimers Disease (AD).png')

        # plt.show()        
    except Exception as e:
        logger.error(f"Error plotting correlation matrice {e}")

def main():
    folder_path = "data/preprocessed_filtered"
    raw_data_labelled = pd.read_csv("data/raw_labelled_data.csv")
    plot_frequency_domain(raw_data_labelled)
    plot_time_frequency_domain(raw_data_labelled)
    
    df_eeg = load_eeg_data(folder_path)
    plot_correlation_matrix(df_eeg)

if __name__ == "__main__":
    main()
