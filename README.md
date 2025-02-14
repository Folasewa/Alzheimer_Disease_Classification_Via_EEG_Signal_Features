
## Data Science Final Workshop Project 

## Diagnosis of Alzheimer's Disease via Resting State EEG - Integration of Spectrum, Complexity and Synchronization Signal Features

### Project Overview

#### This project aims to develop a machine learning pipeline for classifying Alzheimer's Disease (AD) using EEG signal features. The pipeline consists of preprocessing EEG data, extracting spectral, complexity, and synchronization metrics, and training multiple classification models.

### Objectives

#### 1. Process raw EEG data from cognitively normal (CN) and Alzheimer's disease (AD) subjects.

#### 2. Extract relevant EEG biomarkers from spectral, complexity, and synchronization domains.

#### 3. Train and evaluate machine learning models for classifying AD vs CN.

#### 4. Ensure reproducibility by providing a structured pipeline and automation scripts.


### Key Assumptions and Hypothesis

#### 1. EEG signals contain distinct frequency and complexity patterns that differentiate AD from CN.

#### 2. Functional connectivity and synchronization metrics can capture brain network dysfunctions associated with AD.

#### 3. Machine learning models can accurately classify AD vs CN when trained on carefully extracted features.

### Project Structure

```
EEG_Alzheimer_Classification/
│-- data/                  # Dataset directory (original + generated feature datasets)
│   ├── raw_dataset/        # EEG dataset (downloaded from OpenNeuro)
│   ├── filtered_subjects/  # Filtered EEG files
│   ├── filtered_preprocessed/  # Preprocessed EEG files
│   ├── epochs_overlap/     # Extracted EEG epochs
│   ├── spectrum_features.csv    # Spectrum feature dataset
│   ├── complexity_features.csv  # Complexity feature dataset
│   ├── synchronization_features.csv  # Synchronization dataset
│   ├── final_dataset.csv   # Combined dataset for classification
│
│-- model/                    # classification model directory 
│   ├── dt_model.pkl          # saved decision tree model
│   ├── rf_model.pkl          # saved random forest model
│   ├── svm_model.pkl         # saved support vector machine model
│   ├── lightgbm_model.pkl    # saved lightgbm model

│-- plots/                    # visualization plots directory 
│   ├── frequency_domain.png    # saved frequency domain plots for both AD and CN
│   ├── time_frequency_domain.png  # saved time-frequency domain plots for both AD and CN
│   ├── correlation_matrix.png        # saved connectivity matrix for both AD and CN
│
│-- reports/                    # report of my unit and integration test
│   ├── report.html    # html file of the test report

│-- src/                            # Python scripts for EEG processing and modeling
│   ├── preprocessing.py            # EEG Preprocessing (filtering, noise removal, ICA)
│   ├── epoch_extraction.py         # Extracting 4s epochs with 50% overlap
│   ├── spectrum_metrics_extraction.py  # Extracting frequency-domain features (PSD, power ratios)
│   ├── complexity_metrics_extraction.py  # Extracting entropy-based complexity metrics
│   ├── complexity_preprocessing_optimization #post-cleaning on the extracted complexity metrics
│   ├── synchronization_metrics_extraction.py  # Extracting brain network synchronization metrics
│   ├── classification_model.py    # Merging features, training classifiers, evaluation
│   ├── plot_viz.py               #plots frequency and time-frequency domain of AD vs CN
│   ├── statistical_test.py       #Compute independent t-tests between CN and AD groups
│   ├── logger.py                #Logging module for debugging and monitoring
│   ├── main.py                #This script runs the entire EEG pipeline from preprocessing to statistical test


│-- tests/                                      # Python scripts for performing unit and integration tests
│   ├── test_classification_model.py            # unit and integration testing for the classification pipeline
│   ├── test_complexity_metrics_extraction.py   # unit and integration testing for the complexity metrics extraction
│   ├── test_epoch_extraction.py  # unit and integration testing for the epoch extraction
│   ├── test_preprocesing.py  # unit an dintegration testing for EEG preprocessing
│   ├── test_spectrum_metrics_extraction.py #unit and integration testing for spectrum metrics extraction
│   ├── test_synchronization_metrics_extraction.py  # unit and integration testing for synchronization metrics extraction
│   
│
│-- bash.sh                 # Bash script for dataset setup and folder structure
│-- my_project.code-workspace     # VS Code workspace settings
│-- pyproject.toml                # Project metadata and dependencies
│-- README.md               # Project documentation
│-- requirements.txt         # Required dependencies for Python environment
│-- tox.ini         # Automated testing configuration
```

##  Key Stages of the Project
### A. Data Import & Setup

#### -Dataset: EEG data is obtained from OpenNeuro: 
[https://openneuro.org/datasets/ds004504/versions/1.0.2/download#](https://openneuro.org/datasets/ds004504/versions/1.0.2/download#)

#### -Bash Script (bash.sh) automates dataset structure setup.

#### -Options for manual download or DataLad download are included.

### B. EEG Preprocessing (preprocessing.py)
#### -Band-pass filtering (0.5 - 45 Hz) to remove unwanted noise.

#### -Independent Component Analysis (ICA) for artifact removal.

#### -Re-referencing EEG channels using an average reference.

#### -Artifact Subspace Reconstruction (ASR) to exclude high-amplitude segments.

#### -Preprocessed files are saved in .fif format.

### C. Epoch Extraction (epoch_extraction.py)
#### -Extracts 4-second non-overlapping epochs with 50% overlap.
#### -Saves each epoch as an individual .npy file for further feature extraction.

### D Feature Extraction

#### D1 Spectral Metrics (spectrum_metrics_extraction.py)
##### -Time domain metrics:  Mean, Variance and Interquartile Range computed

##### -Power Spectral Density (PSD) computed using Welch’s method.

##### -Band-specific power values (Delta, Theta, Alpha, Beta, Gamma).

##### -Relative Band Power (RBP) normalizes each band’s power against total power.

#### D2 Complexity Metrics (complexity_metrics_extraction.py)

##### -Entropy-based metrics: Approximate Entropy (ApEn), Sample Entropy (SampEn), Permutation Entropy (PermEn).

##### -Raw complexity dataset undergoes additional cleaning to remove NaNs, ensuring compatibility with classifiers.

#### D3 Synchronization Metrics (synchronization_metrics_extraction.py)

##### -Functional connectivity analysis using Pearson correlation and/or Phase Locking Value (PLV).

##### -Graph-based features: Clustering Coefficient, Characteristic Path Length, Global Efficiency, Small-Worldness.

##### -Thresholding strongest 60% of connections to construct binary brain networks.


### E. Feature Merging & Classification (classification_model.py)

#### -Feature datasets (spectral, complexity, synchronization) are merged.

#### -Subjects are labeled (AD = 1, CN = 0).

#### Data cleaning: removing missing values and infinite values.

### Machine Learning Models:

   ##### -Decision Tree

   ##### -Random Forest

   ##### -Support Vector Machine (SVM)

   ##### -LightGBM

### Cross-validation using GroupShuffleSplit to prevent subject overlap between train/test sets.

### Performance Metrics:

   ##### -Accuracy

   ##### -Sensitivity (Recall for AD patients)

   ##### -Specificity (Correctly classifying CN individuals)

### Statistical Test:

   ##### -Independent t-test


## Dataset Description & Source

#### Dataset: [https://openneuro.org/datasets/ds004504/versions/1.0.2/download#](https://openneuro.org/datasets/ds004504/versions/1.0.2/download#)

#### Link to Paper:https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2023.1288295/full#ref18

#### EEG Data:

   #### -65 subjects (36 AD, 29 CN)

   #### -Eyes-closed resting-state EEG

   #### -500 Hz sampling rate

   #### -Electrode placement: 19 scalp electrodes (10-20 system)

#### Extracted Features:

   #### -Time-domain features (mean, variance, IQR), Power Spectral Density (PSD), Relative Band Power

   #### -Approximate Entropy, Sample Entropy, Permutation Entropy

   #### -Functional connectivity metrics and graph theory-based synchronization features.


## Instructions to Run the Project

#### 1. Install Dependencies

`pip install -r requirements.txt`

#### 2. Setup Dataset & Folder Structure

`bash bash.sh`

3. #### Run Individual Steps

#### Preprocessing: 
`python -m run_preprocessing`

#### Epoch Extraction: 
`python -m run_epoch_extraction`

#### Complexity Metrics: 
`python -m run_complexity`

#### Complexity Optimization: 
`python -m run_complexity_optimization`

#### Spectral Metrics: 
`python -m run_spectrum`

#### Synchronization Metrics: 
`python -m run_synchronization`

#### Classification Model: 
`python -m run_classification_model`

#### Visualization: 
`python -m run_plot_visualization`

#### Statistical Analysis: 
`python -m run_statistical_analysis`

#### 4. Run the Complete Pipeline

`python main.py`
#### 5. Run the test
`tox`

#### 6. Expected Output:

   #### -Preprocessed EEG files saved in filtered_preprocessed/.

   #### -Extracted epochs saved in epochs_overlap/.

   #### -Feature datasets: spectrum_features.csv, complexity_features.csv, synchronization_features.csv.

   #### -Final classification results printed with accuracy, sensitivity, specificity.

