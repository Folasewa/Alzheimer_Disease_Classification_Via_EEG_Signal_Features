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
                    frequency_bands[i], fontsize=14, fontweight="bold", ha="center")
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=16)
        ax.set_ylabel("Amplitude (µV²/Hz)", fontsize=16)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
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
        band_edges = [0.5, 4, 8, 13, 25, 45]
        colors = ["#DDA0DD", "#FFD700", "#90EE90", "#D8BFD8", "#ADD8E6"] 

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
        fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

        # Plot CN and AD spectra
        plot_group_psd(axes[0], smooth_freq, cn_spline, band_edges, frequency_bands, colors,
                    "Frequency-Domain Spectrum (CN)", "blue")
        plot_group_psd(axes[1], smooth_freq, ad_spline, band_edges, frequency_bands, colors,
                    "Frequency-Domain Spectrum (AD)", "red")

    except Exception as e:
        logger.error(f"Error plotting frequency-domain spectrum {e}")

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
                    subject_id = int(filename.split("-")[1].split("_")[0])
                except (ValueError, IndexError) as e:
                    logger.error(f"Skipping file due to unexpected name format: {filename} | Error: {e}")
                    continue
                file_path = os.path.join(folder_path, filename)
                # Load EEG data
                raw = mne.io.read_raw_fif(file_path, preload=True)
                # Pick only EEG channels
                eeg_channels = raw.pick_types(eeg=True).ch_names
                #compute mean psd
                psd = raw.compute_psd(method='welch', fmin=1, fmax=40, n_fft=256, n_overlap=128)
                psd_values = psd.get_data()
                eeg_psd_mean = np.mean(psd_values, axis=1)  
                # Append to dataset
                eeg_data_list.append(eeg_psd_mean)
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
        df_ad = df_eeg[df_eeg["label"] == 1].drop(columns=["label"])
        df_cn = df_eeg[df_eeg["label"] == 0].drop(columns=["label"])
        # Compute correlation matrices
        corr_ad = df_ad.corr()
        corr_cn = df_cn.corr()
        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sns.set(font_scale=1.2)
        plt.tight_layout()
        # Plot for Cognitively Normal (CN)
        sns.heatmap(corr_cn, ax=axes[0], cmap="jet", annot=False, fmt=".2f", cbar=True)
        axes[0].set_title("Correlation Matrix - Cognitively Normal (CN)", fontsize=18)
        axes[0].set_xlabel("Electrode", fontsize=18)
        axes[0].set_ylabel("Electrode", fontsize=18)
        axes[0].tick_params(axis='both', labelsize=14)
        #plt.savefig('plots/frequency_domain_Cognitively Normal (CN).png')
        # Plot for Alzheimer's Disease (AD)
        sns.heatmap(corr_ad, ax=axes[1], cmap="jet", annot=False, fmt=".2f", cbar=True)
        axes[1].set_title("Correlation Matrix - Alzheimer's Disease (AD)", fontsize=18)
        axes[1].set_xlabel("Electrode", fontsize=18)
        axes[1].set_ylabel("Electrode", fontsize=18)
        axes[1].tick_params(axis='both', labelsize=14)
        plt.savefig('plots/correlation_matrix_Cognitively Normal (CN)_Alzheimers Disease (AD).jpeg')
    except Exception as e:
        logger.error(f"Error plotting correlation matrice {e}")

def compute_group_stats(features, cn_group, ad_group):
    """
    Computes the median and variance for the specified frequency-domain features.
    """
    try:
        cn_median = [cn_group[[col for col in cn_group.columns if feature in col]].median().median() for feature in features]
        ad_median = [ad_group[[col for col in ad_group.columns if feature in col]].median().median() for feature in features]
        cn_variance = [cn_group[[col for col in cn_group.columns if feature in col]].var().median() for feature in features]
        ad_variance = [ad_group[[col for col in ad_group.columns if feature in col]].var().median() for feature in features]
        return cn_median, ad_median, cn_variance, ad_variance
    except Exception as e:
        logger.error(f"Error in computing stats {e}")

def bar_plot_frequency_domain_features(raw_data_labelled):
    """
    Plots a barplot of frequency-domain comparison between CN and AD groups.
    """
    try:
        # Define feature categories
        freq_domain_features = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
        cn_group = raw_data_labelled[raw_data_labelled['label'] == 0]
        ad_group = raw_data_labelled[raw_data_labelled['label'] == 1]
        # Compute stats using median & variance
        cn_median, ad_median, cn_variance, ad_variance = compute_group_stats(freq_domain_features, cn_group, ad_group)
        # Define category labels
        freq_labels = ["Delta PSD", "Theta PSD", "Alpha PSD", "Beta PSD", "Gamma PSD"]
        bar_width = 0.2
        x_freq = np.arange(len(freq_labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        #Barplot
        ax.bar(x_freq - bar_width/2, cn_median, yerr=cn_variance, capsize=5, color="purple", alpha=0.6, width=bar_width, label="CN")
        ax.bar(x_freq + bar_width/2, ad_median, yerr=ad_variance, capsize=5, color="green", alpha=0.6, width=bar_width, label="AD")

        ax.set_xticks(x_freq)
        ax.set_xticklabels(freq_labels, fontsize=12, ha="center")
        ax.set_title("Frequency-domain Features", fontsize=16, fontweight="bold", loc="center")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/frequency_domain-bar.jpeg')
    except Exception as e:
        logger.error(f"Error plotting the bar graph {e}")

def main():
    folder_path = "data/preprocessed_filtered"
    raw_data_labelled = pd.read_csv("data/raw_labelled_data.csv")
    plot_frequency_domain(raw_data_labelled)
    bar_plot_frequency_domain_features(raw_data_labelled)
    df_eeg = load_eeg_data(folder_path)
    plot_correlation_matrix(df_eeg)

if __name__ == "__main__":
    main()
