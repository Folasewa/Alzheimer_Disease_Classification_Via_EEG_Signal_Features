from src.classification_model import apply_label, data_cleaning, data_splitting, evaluate_model
from src.classification_model import train_evaluate_decision_tree, train_evaluate_random_forest
from src.classification_model import train_evaluate_svm, train_evaluate_lightgbm, cross_validate_models
import pandas as pd
import numpy as np
import pickle as pkl

def test_apply_label_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    assert raw_data_labelled["label"].tolist()[:2] == [1, 0]

def test_apply_label_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    assert raw_data_labelled.empty

def test_data_cleaning_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled, x, y = data_cleaning(raw_data_labelled)
    assert len(x)
    assert len(y)
    assert not  x.isna().sum().sum()
    assert not np.isinf(x).sum().sum()
    assert not raw_data_labelled.empty

def test_data_cleaning_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled, x, y = data_cleaning(raw_data_labelled)
    assert not len( x)
    assert not len(y)

def test_data_splitting_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled, x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    assert len( x_train), len( x_test)
    assert len(y_train), len(y_test)

def test_data_splitting_neg():
    raw_data = pd.DataFrame([])
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled, x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    assert x_train.empty, x_test.empty
    assert not y_train, not y_test

def test_evaluate_model_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    with open('tests/test_model/dt_model.pkl', 'rb') as file:
        model = pkl.load(file)
    accuracy, specificity, sensitivity = evaluate_model(model,  x_test, y_test)
    assert accuracy > 0
    assert specificity > 0
    assert sensitivity > 0

def test_evaluate_model_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    model = "/model"
    accuracy, specificity, sensitivity = evaluate_model(model,  x_test, y_test)
    assert not accuracy
    assert not specificity
    assert not sensitivity

def test_train_evaluate_decision_tree_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    dt_model, accuracy, sensitivity, specificity = train_evaluate_decision_tree(x_train,  x_test, y_train, y_test)
    assert dt_model
    assert accuracy > 0
    assert sensitivity > 0
    assert specificity > 0

def test_train_evaluate_decision_tree_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting(x, y)
    dt_model, accuracy, sensitivity, specificity = train_evaluate_decision_tree( x_train,  x_test, y_train, y_test)
    assert not dt_model
    assert not accuracy
    assert not sensitivity
    assert not specificity

def test_train_evaluate_random_forest_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    rf_model, accuracy, sensitivity, specificity = train_evaluate_random_forest( x_train,  x_test, y_train, y_test)
    assert rf_model
    assert accuracy > 0
    assert sensitivity > 0
    assert specificity > 0

def test_train_evaluate_random_forest_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    rf_model, accuracy, sensitivity, specificity = train_evaluate_random_forest(x_train,  x_test, y_train, y_test)
    assert not rf_model
    assert not accuracy
    assert not sensitivity
    assert not specificity

def test_train_evaluate_svm_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    svm_model, accuracy, sensitivity, specificity = train_evaluate_svm( x_train,  x_test, y_train, y_test)
    assert svm_model
    assert accuracy > 0
    assert sensitivity > 0
    assert specificity > 0

def test_train_evaluate_svm_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    svm_model, accuracy, sensitivity, specificity = train_evaluate_svm( x_train,  x_test, y_train, y_test)
    assert not svm_model
    assert not accuracy
    assert not sensitivity
    assert not specificity

def test_train_evaluate_lightgbm_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    lgb_model, accuracy, sensitivity, specificity = train_evaluate_lightgbm( x_train,  x_test, y_train, y_test)
    assert lgb_model
    assert accuracy > 0
    assert sensitivity > 0
    assert specificity > 0

def test_train_evaluate_lightgbm_neg():
    raw_data = pd.DataFrame()
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    x_train,  x_test, y_train, y_test = data_splitting( x, y)
    lgb_model, accuracy, sensitivity, specificity = train_evaluate_lightgbm( x_train,  x_test, y_train, y_test)
    assert not lgb_model
    assert not accuracy
    assert not sensitivity
    assert not specificity

def test_cross_validate_model_pos():
    raw_data = pd.read_csv("tests/test_data/dataset_project_ds_nolabel.csv")
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    dt_metrics, rf_metrics, svm_metrics, lgb_metrics = cross_validate_models( x, y, raw_data, n_splits=5, test_size=0.2, random_state=42)
    assert dt_metrics
    assert rf_metrics
    assert svm_metrics
    assert lgb_metrics

def test_cross_validate_model_neg():
    raw_data = pd.DataFrame([])
    raw_data_labelled = apply_label(raw_data)
    raw_data_labelled,  x, y = data_cleaning(raw_data_labelled)
    dt_metrics, rf_metrics, svm_metrics, lgb_metrics = cross_validate_models( x, y, raw_data, n_splits=5, test_size=0.2, random_state=42)
    assert not dt_metrics
    assert not rf_metrics
    assert not svm_metrics
    assert not lgb_metrics

