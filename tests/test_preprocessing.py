import os
from src.preprocessing import read_data, copy_data, check_noise, write_data
import glob


def test_pos_read_data():
    data_path = "test_data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    data = read_data(data_path)
    assert data is not None

def test_neg_read_data():
    data_path = "test_data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    data = read_data(data_path)
    assert data is None

def test_pos_copy_data():
    source_path = "test_data"
    destination_path = "filtered_subjects"
    os.path.makedirs(destination_path, exist_ok=True)
    copy_data(source_path, destination_path)
    data_dir = glob.glob(os.path.join(destination_path, "*.set"))
    assert os.path.exists(destination_path)
    assert len(data_dir) > 0
    os.removedirs(destination_path)

def test_neg_copy_data():
    source_path = "test_dataz"
    destination_path = "filtered_subjects"
    os.path.makedirs(destination_path, exist_ok=True)
    copy_data(source_path, destination_path)
    data_dir = glob.glob(os.path.join(destination_path, "*.set"))
    assert len(data_dir) == 0
    os.removedirs(destination_path)

def test_pos_check_noise_present():
    dataset = read_data("test_data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 20
    noise_dataset = check_noise(dataset, noise_threshold, n_components)
    assert noise_dataset is not None
    assert dataset != noise_dataset

def test_pos_check_noise_absent():
    dataset = read_data("test_data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 20
    noise_dataset = check_noise(dataset, noise_threshold, n_components)
    assert noise_dataset is not None
    assert dataset == noise_dataset

def test_neg_check_noise():
    dataset = read_data("test_data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 20
    check_noise(dataset, noise_threshold, n_components)
    assert dataset is None

