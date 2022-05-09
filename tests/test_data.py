import pytest

from cover_type_classifier.data import get_dataset
from cover_type_classifier.data import generate_eda
from cover_type_classifier.data import feature_engineering
from click.testing import CliRunner

from typing import Union

Fixture = Union


def test_get_dataset(write_to_file: Fixture[str, str]) -> None:
    """
    Method should return non-empty dataframe.
    """
    data_path, test_path = write_to_file
    X_train, y_train, X_test = get_dataset.get_dataset(data_path, test_path)
    assert X_train.shape != (0, 0)
    assert y_train.shape != (0, 0)
    assert X_test.shape != (0, 0)


@pytest.mark.parametrize(
    "data_path, test_path",
    [("data/external/train_1.csv", "data/external/test.csv")],
)
def test_get_dataset_negative(data_path: str, test_path: str) -> None:
    """
    It fails, when there is no relevant file in directory.
    """
    with pytest.raises(FileNotFoundError) as err:
        get_dataset.get_dataset(data_path, test_path)
    assert "data/external/train_1.csv" in str(err.value)


def test_error_for_invalid_profiler(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        generate_eda.generate_eda,
        [
            "--profiler",
            "test",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--profiler':" in result.output


def test_feature_enginnering(write_to_file: Fixture[str, str]) -> None:
    """
    Method should return non-empty dataframe.
    """
    data_path, test_path = write_to_file
    X_train, y_train, X_test = get_dataset.get_dataset(data_path, test_path)

    X_train_proc, y_train_proc, X_test_proc = feature_engineering.process_data(
        X_train, y_train, X_test, True, True
    )
    assert (
        X_train_proc.shape[1] <= X_train.shape[1]
        and X_test_proc.shape[1] <= X_test.shape[1]
    )
    print(0 <= X_train_proc)
    assert ((0 <= X_train_proc) & (X_train_proc < 2)).all()
    assert ((0 <= X_test_proc) & (X_test_proc < 2)).all()
