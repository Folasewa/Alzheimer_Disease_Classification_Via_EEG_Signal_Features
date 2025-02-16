from src.synchronization_metrics_extraction import connectivity_matrix, threshold_matrix, compute_graph_metrics, process_and_save_sync_epoch
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from src.utilities import checkpoint

def test_connectivity_matrix_pos():
    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    for method in ['pearson' or 'plv']:
        conn_matrix = connectivity_matrix(epoch, method)
        assert conn_matrix is not None

def test_connectivity_matrix_neg():
    epoch_file = "tests/test_data/epochs_overlap/sub-001_epoch-001.npy"
    epoch = np.load(epoch_file)
    method = 'pear'
    conn_matrix = connectivity_matrix(epoch, method)
    assert not len(conn_matrix)

def test_threshold_matrix_pos():
    epoch = np.load("tests/test_data/epochs_overlap/sub-001_epoch-001.npy")
    connectivity = connectivity_matrix(epoch, "pearson")
    print("##################", connectivity)
    threshold_value = 0.8
    adjacency = threshold_matrix(connectivity, threshold_value)
    print("^^^^^^^^^^^^^^^^^^^^^^^", adjacency)
    assert len(adjacency)

def test_threshold_matrix_neg():
    epoch = np.array([])
    connectivity = connectivity_matrix(epoch, "pearson")
    threshold_value = 0.8
    adjacency = threshold_matrix(connectivity, threshold_value)
    assert not len(adjacency)

def test_process_and_save_sync():
     graph_metrics_csv = "tests/test_data/synchronization.csv"
     synchronization_method = "pearson"
     threshold_value = 0.6
     output_folder = "tests/test_data/epochs_overlap"
     files_to_process, processed_set = checkpoint(graph_metrics_csv, output_folder)
     files_to_process = [os.path.join(output_folder, f) for f in files_to_process]
     df_metrics_list = Parallel(n_jobs=-1)(
        delayed(process_and_save_sync_epoch)(file, graph_metrics_csv, synchronization_method, threshold_value)
        for file in files_to_process
    )
     assert len(df_metrics_list) > 0

