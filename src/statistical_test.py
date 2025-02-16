# T-test
import numpy as np
import pandas as pd
import scipy.stats as stats
from logger import setup_logger


logger = setup_logger("logger", "logs.log")

def compute_t_tests(features, cn_group, ad_group):
    """
    Compute independent t-tests between CN and AD groups for the given features.
    """
    try:
        results = []
        for feature in features:
        # Select columns that contain the feature name
            cn_values = cn_group[[col for col in cn_group.columns if feature in col]].mean().dropna()
            ad_values = ad_group[[col for col in ad_group.columns if feature in col]].mean().dropna()
            if len(cn_values) > 1 and len(ad_values) > 1:
                t_stat, p_val = stats.ttest_ind(cn_values, ad_values, equal_var=False)
            else:
                p_val = np.nan
            cn_mean, ad_mean = cn_values.mean(), ad_values.mean()
            trend = "Increase" if ad_mean > cn_mean else "Decrease"
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "Not Significant"
            results.append([feature, cn_mean, ad_mean, trend, p_val, significance])
    except Exception as e:
        logger.error(f"Error computing T-test {e}")
        return pd.DataFrame([], columns=["Feature", "CN Mean", "AD Mean", "Trend", "p-value", "Significance"])

    return pd.DataFrame(results, columns=["Feature", "CN Mean", "AD Mean", "Trend", "p-value", "Significance"])

def print_results(title, df):
    """
    Prints the statistical test results in a readable format.
    """
    logger.info(f"\n### {title} ###\n")
    logger.info(df.to_string(index=False))

def compute_t_test_across(raw_data_labelled):
    """
    Main function to perform t-tests across different EEG feature categories
    """
    try:
        # Define feature categories
        feature_categories = {
            "Time-Domain Metrics": ['mean', 'variance', 'IQR'],
            "Frequency-Domain Metrics": ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma'],
            "Complexity Metrics": ['ApEn', 'PermEn', 'SampEn'],
            "Synchronization Metrics": ['clustering_coefficient', 'characteristic_path_length', 
                                        'global_efficiency', 'small_worldness']
        }

        # Separate CN and AD groups
        cn_group = raw_data_labelled[raw_data_labelled['label'] == 0]
        ad_group = raw_data_labelled[raw_data_labelled['label'] == 1]

        # Compute and print results for each feature category
        for category, features in feature_categories.items():
            results = compute_t_tests(features, cn_group, ad_group)
            print_results(category, results)
    except Exception as e:
        logger.error(f"Error computing t-test across all features {e}")

def main():
    raw_data_labelled = pd.read_csv("data/raw_labelled_data.csv")
    compute_t_test_across(raw_data_labelled)

if __name__ == "__main__":
    main()
