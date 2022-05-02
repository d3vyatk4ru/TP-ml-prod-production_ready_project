from src.data.make_dataset import read_dataset, split_train_val_data
from src.entity.split_params import SplittingParams

def test_read_dataset(path: str):
    data = read_dataset(path)
    assert 297 == len(data)

def test_split_data(path):

    data = read_dataset(path)

    splitting_params = SplittingParams(random_state=42, val_size=0.1)

    train, test = split_train_val_data(data, splitting_params)

    assert 267 == len(train)
    assert 30 == len(test)

