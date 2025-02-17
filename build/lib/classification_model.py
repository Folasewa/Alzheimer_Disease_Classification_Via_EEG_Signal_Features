from logger import setup_logger
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from statistical_test import compute_t_test_across

logger = setup_logger("logger", "logs.log")

def assign_label(subject_id):

    """
    Assign label 1 for AZ subjects based on their subject id
    Assign label 0 for CN subjects based on their subject_id
    
    """
    try:
        subject_num = int(subject_id.split('-')[1])
        if subject_num in range(1,37):
            return 1
        elif subject_num in range(37, 66):
            return 0
        else:
            return None
    except Exception as e:
        logger.error(f"Error assigning label {e}")
        return None

def apply_label(raw_data):
    """
    This function applies label assignment on raw data
    """
    try:
        raw_data["label"] = raw_data["subject_id"].apply(assign_label)
        raw_data_labelled = raw_data
    except Exception as e:
        logger.error(f"Error applying label: {e}")
        raw_data_labelled = pd.DataFrame([])
    return raw_data_labelled

def data_cleaning(raw_data_labelled):

    """
    This function cleans the data, 
    selects the numeric columns and removes row with inf/-inf/nan
    """
    try:
        numeric_cols = raw_data_labelled.select_dtypes(include=[np.number]).columns

        # Remove rows with inf or -inf in numeric columns
        raw_data_labelled = raw_data_labelled[~np.isinf(raw_data_labelled[numeric_cols]).any(axis=1)]

        # Remove rows with NaN in the dataset
        raw_data_labelled = raw_data_labelled.dropna()
        raw_data_labelled.to_csv('data/raw_labelled_data.csv', index=False)

        X = raw_data_labelled.drop(["label", "subject_id", "epoch_number"], axis = 1)
        y = raw_data_labelled["label"]
    except Exception as e:
        raw_data_labelled = pd.DataFrame([])
        X = pd.DataFrame([])
        y = []
        logger.error(f"Error cleaning data: {e}")

    return raw_data_labelled, X, y

def data_splitting(X, y):
    """
    This function splits the data into train and test sets
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        X_train, X_test, y_train, y_test = pd.DataFrame([]), pd.DataFrame([]), [], []
        logger.error("Error splitting the dataset {e}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    This function evaluates the model's performance
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Printing these metrics
        logger.info("Model Performance Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating the model {e}")
        accuracy, specificity, sensitivity = None, None, None
    return accuracy, specificity, sensitivity

# Functions for model building
def train_evaluate_decision_tree(X_train, X_test, y_train, y_test):
    """
    This function trains and evaluates the decision tree model
    """
    try:
        logger.info("\n.....training and evaluating with Decision Tree....")
        dt_model = DecisionTreeClassifier(random_state=42,
                                        class_weight='balanced',
                                        max_depth=5, min_samples_split=5, min_samples_leaf=5)
        dt_model.fit(X_train, y_train)
        accuracy, sensitivity, specificity = evaluate_model(dt_model, X_test, y_test)
    except Exception as e:
        logger.error(f"Error training and evaluating the decision tree model {e}")
        dt_model = None
        accuracy, sensitivity, specificity = None, None, None
    return dt_model, accuracy, sensitivity, specificity

def train_evaluate_random_forest(X_train, X_test, y_train, y_test):
    """
    This function trains and evaluates the Random Forest model
    """
    try:
        logger.info("\n ... training and evaluating with Random Forest ...")
        rf_model = RandomForestClassifier(n_estimators= 100,
                                        max_depth=10,
                                        min_samples_split=5,
                                        min_samples_leaf=2,
                                        class_weight='balanced',
                                        random_state=42)
        rf_model.fit(X_train, y_train)
        accuracy, sensitivity, specificity = evaluate_model(rf_model, X_test, y_test)
    except Exception as e:
        logger.error(f"Error training and evaluating the random forest model {e}")
        rf_model = None
        accuracy, sensitivity, specificity = None, None, None
    return rf_model, accuracy, sensitivity, specificity

def train_evaluate_svm(X_train, X_test, y_train, y_test):
    """
    This function trains and evaluates the SVM model
    """
    try:
        logger.info("\n...training and evaluating with svm ...")
        svm_model = SVC(random_state=42, class_weight='balanced', kernel="linear")
        svm_model.fit(X_train, y_train) 
        accuracy, sensitivity, specificity = evaluate_model(svm_model, X_test, y_test)
    except Exception as e:
        logger.error(f"Error training and evaluating the SVM model {e}")
        svm_model= None
        accuracy, sensitivity, specificity = None, None, None
    return svm_model, accuracy, sensitivity, specificity

def train_evaluate_lightgbm(X_train, X_test, y_train, y_test):
    """
    This function trains and evaluates the lightgbm model
    """
    try:
        logger.info("\n...training and evaluating with lightgbm...")
        lgb_model = lgb.LGBMClassifier(class_weight="balanced",random_state=42)
        lgb_model.fit(X_train, y_train)
        accuracy, sensitivity, specificity = evaluate_model(lgb_model, X_test, y_test)
    except Exception as e:
        logger.error(f"Error training and evaluating the lightgbm model {e}")
        lgb_model = None
        accuracy, sensitivity, specificity = None, None, None
    return lgb_model, accuracy, sensitivity, specificity

def initialize_metrics_dict():
    """
    This function initializes a dictionary to store the performance metrics
    """
    return {'accuracy': [], 'sensitivity': [], 'specificity': []}

def fill_metrics_dict(metrics_dict, model, accuracy, sensitivity, specificity):
    """
    This function updates the metrics dictionary with new evaluation results.
    """
    try:
        metrics_dict['accuracy'].append(accuracy)
        metrics_dict['sensitivity'].append(sensitivity)
        metrics_dict['specificity'].append(specificity)
    except Exception as e:
        logger.error(f"Error updating the metrics dictionary {e}")
    return metrics_dict

# CROSS VALIDATION USING GroupShuffleSplit
def cross_validate_models(X, y, raw_data, n_splits=5, test_size=0.2, random_state=42):
    """
    This function performs cross validation using GroupShuffleSplit
    """
    try:
        # Initialize dictionaries to store metrics for each model
        dt_metrics = initialize_metrics_dict()
        rf_metrics = initialize_metrics_dict()
        svm_metrics = initialize_metrics_dict()
        lgb_metrics = initialize_metrics_dict()

        groups = raw_data['subject_id']

        # GroupShuffleSplit instance
        group_splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        # Iterate through the splits
        for fold_idx, (train_idx, test_idx) in enumerate(group_splitter.split(X, y, groups=groups)):
            # Split data into train and test sets
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Ensure both classes are present in the test set
            if len(y_test.unique()) < 2:
                logger.info(f"Skipping Fold {fold_idx + 1}: Test set does not contain both classes.")
                continue
            logger.info(f"\n--- Fold {fold_idx + 1} ---")

            # Initialize metrics dictionaries for each model
            dt_model, accuracy, sensitivity, specificity = train_evaluate_decision_tree(X_train, X_test, y_train, y_test)
            dt_metrics = fill_metrics_dict(dt_metrics, dt_model, accuracy, sensitivity, specificity)

            # Random Forest
            rf_model = train_evaluate_random_forest(X_train, X_test, y_train, y_test)
            rf_metrics = fill_metrics_dict(rf_metrics, rf_model, accuracy, sensitivity, specificity)
            # SVM
            svm_model = train_evaluate_svm(X_train, X_test, y_train, y_test)
            svm_metrics = fill_metrics_dict(svm_metrics, svm_model, accuracy, sensitivity, specificity)
            #lightgbm
            lgb_model = train_evaluate_lightgbm(X_train, X_test, y_train, y_test)
            lgb_metrics = fill_metrics_dict(lgb_metrics, lgb_model, accuracy, sensitivity, specificity)
    except Exception as e:
        dt_metrics, rf_metrics, svm_metrics, lgb_metrics = {}, {}, {}, {}
        logger.error(f"Error performing cross-validation {e}")
    return dt_metrics, rf_metrics, svm_metrics, lgb_metrics

def show_average_metrics(model_name, metrics_dict):
    """
    This function prints average metrics across all folds
    """
    logger.info("\n--- Average Metrics Across All Folds ---")
    logger.info(f"Model: {model_name}")
    logger.info(f"Average Accuracy: {sum(metrics_dict['accuracy']) / len(metrics_dict['accuracy']):.4f}")
    logger.info(f"Average Sensitivity: {sum(metrics_dict['sensitivity']) / len(metrics_dict['sensitivity']):.4f}")
    logger.info(f"Average Specificity: {sum(metrics_dict['specificity']) / len(metrics_dict['specificity']):.4f}")



def average_metrics(dt_metrics, rf_metrics, svm_metrics, lgb_metrics):
    """
    This function prints Average Metrics for Each Model
    """

    # Decision Tree
    show_average_metrics("Decision Tree", dt_metrics)
    # SVM
    show_average_metrics("SVM", svm_metrics)
    # Lightgbm
    show_average_metrics("LightGBM", lgb_metrics)
    # Random Forest
    show_average_metrics("Random Forest", rf_metrics)

def save_model(model, path):
    """
    This function saves a model to path
    """
    try:
        with open(path, "wb") as file:
            pkl.dump(model, file)
    except Exception as e:
        logger.error("Error saving model.")


def classification_pipeline(spectrum_raw_data, complexity_data, sync_data):
    """
    This function executes the full pipeline for the classification model
    """
    try:
        combined_data = pd.merge(spectrum_raw_data, complexity_data, on = ['subject_id', 'epoch_number'])
        raw_data = pd.merge(combined_data, sync_data, on = ['subject_id', 'epoch_number'])
        raw_data_labelled = apply_label(raw_data)
        raw_data_labelled, X, y = data_cleaning(raw_data_labelled)
        X_train, X_test, y_train, y_test = data_splitting(X, y)

        dt_model, accuracy, sensitivity, specificity = train_evaluate_decision_tree(X_train, X_test, y_train, y_test)
        save_model(dt_model, "model/dt_model.pkl")
        rf_model, accuracy, sensitivity, specificity = train_evaluate_random_forest(X_train, X_test, y_train, y_test)
        save_model(rf_model, "model/rf_model.pkl")
        svm_model, accuracy, sensitivity, specificity = train_evaluate_svm(X_train, X_test, y_train, y_test)
        save_model(svm_model, "model/svm_model.pkl")
        light_model, accuracy, sensitivity, specificity = train_evaluate_lightgbm(X_train, X_test, y_train, y_test)
        save_model(light_model, "model/light_model.pkl")

        dt_metrics, rf_metrics, svm_metrics, lgb_metrics = cross_validate_models(X, y, raw_data_labelled, n_splits=5, test_size=0.2, random_state=42)
        average_metrics(dt_metrics, rf_metrics, svm_metrics, lgb_metrics)
        compute_t_test_across(raw_data_labelled)
        return True
    except Exception as e:
        logger.error(f"Error running the classification pipeline {e}")
        return None

def main():
    spectrum_raw_data = pd.read_csv('data/spectrum.csv')
    complexity_data = pd.read_csv('data/complexity_csv_file.csv')
    sync_data = pd.read_csv('data/synchronization.csv')
    # Combining all the csv files to form one dataset
    main(spectrum_raw_data, complexity_data, sync_data)


if __name__ == "__main__":
    main()
