from logger import setup_logger
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import networkx as nx
from utilities import checkpoint

logger = setup_logger("logger", "logs.log")

def connectivity_matrix(epoch, method):
    """
    Compute the functional connectivity matrix for an epoch

    Parameters:
    -epoch: 2D numpy array (n_channels, n_samples_per_epoch)
    -method: synchronization method which can be pearson or phase locking value (plv)

    Returns:
    -connectivity_matrix: 2D numpy (n_channels, n_channels)

    """
    try:
        n_channels = epoch.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                    if method == 'pearson': #for pearson method
                        connectivity_matrix[i, j] = np.corrcoef(epoch[i, :], epoch[j, :])[0,1]
                    elif method == 'plv':
                        phase_i = np.angle(np.fft.fft(epoch[i, :])) #extract phases using fast fourier transform
                        phase_j = np.angle(np.fft.fft(epoch[j, :]))
                        phase_diff = phase_i - phase_j # instantaneous phase difference
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))#plv
                        connectivity_matrix[i, j] = plv
                    else:
                        raise ValueError("Invalid method of synchronization, choose either pearson or plv")
    except Exception as e:
            logger.error(f"Error computing the connectivity matrix {e}")
            connectivity_matrix = np.array([])

    return connectivity_matrix

def threshold_matrix(connectivity_matrix, threshold ):
    """
    Threshold the functional connectivity matrix

    Parameters:
    -connectivity_matrix : 2D Numpy array (n_channels, n_channels)
    -threshold: proportion of strongest connection to keep (e.g 0.1 for 10%)

    Returns:
    -adjacency_matrix: 2D numpy array , binary matrix (0, 1)

    """

    try:
        n_channels = connectivity_matrix.shape[0]
        # Flatten and sort connections
        sorted_values = np.sort(connectivity_matrix.flatten())[::-1]
        cutoff = sorted_values[int(threshold * n_channels * n_channels)]

        # Apply the threshold
        adjacency_matrix = (connectivity_matrix >=cutoff).astype(int)

        # Remove self connections
        adjacency_matrix = np.fill_diagonal(adjacency_matrix, 0)
    except Exception as e:
        logger.error(f"Error thresholding the matrix {e}")
        adjacency_matrix = np.array([])

    return adjacency_matrix

# Step 3: Build the Graph and compute the graph metrics

def compute_graph_metrics(adjacency_matrix, n_swaps = None):
    """
    Compute the graph metrics from the adjacency matrix

    Parameters:
    -adjacency-matrix: 2D numpy array, binary matrix (0s and 1s)
    -n_swaps: nu,ber of edge swaps for null model. If none, defaults to 10 times the number of edges in the observed graphs

    Returns:
    -metrics: dict containing clustering coefficient, characteristic path length, global efficiency and small worldness

    """
    try:
        G = nx.from_numpy_array(adjacency_matrix)

        # Compute the graph metrics
        clustering_coefficient = nx.average_clustering(G)

        try:
            path_length = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            path_length = np.inf #handle disconnected graphs

        global_efficiency = nx.global_efficiency(G)

        # smallworldness = standardized clustering coefficients /  standardized characteristic path length

        # Determine n_swaps if not provided
        n_edges = len(G.edges)
        if n_swaps is None:
            n_swaps = 10 * n_edges #default to 10 times th enumber of edges

        # Create Null Model(shuffle edges)
        G_Null = G.copy()
        nx.double_edge_swap(G_Null, nswap = n_swaps, max_tries=n_swaps*10)

        # Null model metrics
        clustering_coefficient_null = nx.average_clustering(G_Null)

        try:
            path_length_null = nx.average_shortest_path_length(G_Null)
        except nx.NetworkXError:
            path_length_null = np.inf

        # Compute the standardized metrics (Gamma and Delta)

        gamma = clustering_coefficient / clustering_coefficient_null if clustering_coefficient_null > 0 else 0
        delta = path_length / path_length_null if path_length_null > 0 else np.inf

        small_worldness = gamma/delta if delta > 0 else 0

        # Return all the metrics

        metrics = {
            "clustering_coefficient": clustering_coefficient,
            "characteristic_path_length": path_length,
            "global_efficiency": global_efficiency,
            "small_worldness": small_worldness,

        }
        return metrics
    except Exception as e:
        logger.error(f"Error computing graph metrics {e}")
        return {
            "clustering_coefficient": np.nan,
            "characteristic_path_length": np.nan,
            "global_efficiency": np.nan,
            "small_worldness": np.nan,

        }

# Moving on the last part of feature extraction which is Synchronization Feature

def process_and_save_sync_epoch(file_path, csv_file, synchronization_method, threshold_value):
    """
    Process a single epoch file, compute graph metrics, and write to a CSV file.

    Parameters:
    - file_path: Path to the .npy epoch file.
    - csv_file: Path to the CSV file to save results incrementally.
    - synchronization_method: Synchronization method for connectivity matrix ('pearson' or 'plv').
    - threshold_value: Proportion of strongest connections to keep in adjacency matrix.
    """
    try:
        # Load the epoch data
        epoch = np.load(file_path)

        # Step 1: Compute the connectivity matrix
        connectivity = connectivity_matrix(epoch, synchronization_method)

        # Step 2: Apply thresholding to create the adjacency matrix
        adjacency = threshold_matrix(connectivity, threshold_value)

        # Step 3: Compute graph metrics from the adjacency matrix
        metrics = compute_graph_metrics(adjacency)

        # Add metadata (subject ID and epoch number)
        file_name = os.path.basename(file_path)
        subject_id = file_name.split('_')[0]  # Extract subject ID
        epoch_number = file_name.split('_')[1].replace('epoch-', '').replace('.npy', '')  # Extract epoch number
        metrics['subject_id'] = subject_id
        metrics['epoch_number'] = int(epoch_number)

        # Convert metrics to a DataFrame
        df_metrics = pd.DataFrame([metrics])

        # Step 4: Write to the CSV file incrementally (append mode)
        write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0  # Check if file is empty
        with open(csv_file, 'a') as f:
            df_metrics.to_csv(f, header=write_header, index=False)

        logger.info(f"Processed and saved: {file_name}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        df_metrics = pd.DataFrame([])
    return df_metrics

def main():
    # Parameters
    synchronization_method = "pearson"  # for pearson method
    threshold_value = 0.6  # 60% strongest connections
    graph_metrics_csv = "data/synchronization.csv"
    output_folder = "data/epochs_overlap"  # Path to the folder containing .npy epoch files

    # Main processing
    files_to_process, processed_set = checkpoint(graph_metrics_csv, output_folder)
    files_to_process = list(map(lambda x: output_folder + "/" + x, files_to_process))
    # print(files_to_process, processed_set)
    # Process remaining epochs in parallel
    df_metrics = Parallel(n_jobs=-1)(
        delayed(process_and_save_sync_epoch)(file, graph_metrics_csv, synchronization_method, threshold_value)
        for file in files_to_process
    )

    logger.info(f"All epochs processed and saved to {graph_metrics_csv}.")

if __name__ == "__main__":
    main()
