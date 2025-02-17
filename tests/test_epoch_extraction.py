from src.epoch_extraction import extract_epochs, batch_extract_epochs
from src.preprocessing import read_data

def test_pos_extract_epochs():
    data_path = "tests/test_data/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    raw = read_data(data_path)
    epoch = extract_epochs(raw, epoch_length=4, overlap=0.5)
    assert len(epoch)

def test_neg_extract_epochs():
    data_path = "tests/test_data/suc-001/eeg/sub-001_task-eyesclosed_eeg.set"
    raw = read_data(data_path)
    epoch = extract_epochs(raw, epoch_length=4, overlap=0.5)
    assert not len(epoch)

def test_pos_batch_extract_epochs():
    data_folder_path = "tests/test_data/preprocessed_filtered"
    output_folder_path = "tests/test_data/epochs_overlap"
    epochs_data = batch_extract_epochs(data_folder_path, output_folder_path)
    assert len(epochs_data)

def test_neg_batch__extract_epochs():

    data_folder_path = "/preproce_filtered"
    output_folder_path = "/epochs_overlap"
    epochs_data = batch_extract_epochs(data_folder_path, output_folder_path)

    assert not len(epochs_data)

