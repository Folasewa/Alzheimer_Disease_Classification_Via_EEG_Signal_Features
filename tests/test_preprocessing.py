import os
import shutil
from src.preprocessing import read_data, copy_data, check_noise, write_data
import glob


def test_pos_read_data():
    data_path = "tests/test_data/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    data = read_data(data_path)
    assert data is not None

def test_neg_read_data():
    data_path = "tests/test_data/sub-001/eeg/sub-not-existing-path-001_task-eyesclosed_eeg.set"
    data = read_data(data_path)
    assert data is None

def test_pos_copy_data():
    source_path = "tests/test_data/ds004504/"
    destination_path = "tests/test_data/filtered_subjects"
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path, exist_ok=True)
    copy_data(source_path, destination_path)
    data_dir = glob.glob(os.path.join(destination_path, "*.set"))
    assert os.path.exists(destination_path)
    assert len(data_dir) > 0
    shutil.rmtree(destination_path)

def test_neg_copy_data():
    source_path = "test_dataz"
    destination_path = "filtered_subjects"
    os.makedirs(destination_path, exist_ok=True)
    copy_data(source_path, destination_path)
    data_dir = glob.glob(os.path.join(destination_path, "*.set"))
    assert len(data_dir) == 0
    shutil.rmtree(destination_path)

def test_pos_check_noise_present():
    dataset = read_data("tests/test_data/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 9
    noise_dataset = check_noise(dataset, noise_threshold, n_components)
    assert noise_dataset is not None
    # assert dataset != noise_dataset

def test_pos_check_noise_absent():
    dataset = read_data("tests/test_data/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 9
    noise_dataset = check_noise(dataset, noise_threshold, n_components)
    assert noise_dataset is not None
    assert dataset == noise_dataset

def test_neg_check_noise():
    dataset = read_data("tests/test_data/sub-001/eeg/sub-000001_task-eyesclosed_eeg.set")
    noise_threshold = 1e-6
    n_components = 9
    check_noise(dataset, noise_threshold, n_components)
    assert dataset is None

