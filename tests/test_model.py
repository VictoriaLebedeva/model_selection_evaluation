import pandas as pd 

from typing import Union
from cover_type_classifier.data import get_dataset
from cover_type_classifier.models import train_models

Fixture = Union 

@pytest.fixture
def get_prediction_knn(write_to_file: Fixture[str, str]) -> pd.Series:
    data_path, test_path = write_to_file
    X_train, y_train, X_test = get_dataset.get_dataset(data_path, test_path)
    y_pred = train_models.train(X_train, y_train, X_test, {}, False, 'knn')
    return y_pred


def test_train_knn(write_to_file: Fixture[str, str]) -> None:
    """
    If fails when prediction is not valid.
    """
    data_path, test_path = write_to_file
    X_train, y_train, X_test = get_dataset.get_dataset(data_path, test_path)
    y_pred = train_models.train(X_train, y_train, X_test, {}, False, 'knn')

    assert y_pred.shape[0] == X_test.shape[1]
    assert set(y_pred) == set(y_train)


def test_train_random_forest(write_to_file: Fixture[str, str]) -> None:
    """
    If fails when prediction is not valid.
    """
    data_path, test_path = write_to_file
    X_train, y_train, X_test = get_dataset.get_dataset(data_path, test_path)
    y_pred = train_models.train(X_train, y_train, X_test, {}, False, 'random_forest')

    assert y_pred.shape[0] == X_test.shape[1]
    assert set(y_pred) == set(y_train)


def test_save_prediction_to_file(get_prediction_knn, tmpdir_factory) -> None:
     y_pred = get_prediction_knn
     path = str(tmpdir_factory.mktemp("predictions"))
     pred_path = train_models.write_prediction_to_file(y_pred.index, y_pred, 'knn', path)

     prediction = pd.read_csv(pred_path)

     assert y_pred.shape == prediction['Cover_Type'].shape
     assert y_pred.eq(prediction['Cover_Type'])