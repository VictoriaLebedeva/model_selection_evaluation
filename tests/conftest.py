import pytest
import numpy as np
import pandas as pd

from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def generate_toy_dataset(seed):
    nrows = 1000
    np.random.seed(seed)

    fourty_categories_encoding = pd.get_dummies(
        pd.Series(np.random.randint(0, 41, nrows))
    ).values
    four_categories_encoding = pd.get_dummies(
        pd.Series(np.random.randint(1, 4, nrows))
    ).values
    numerical_data = np.random.rand(nrows, 11)

    X = np.concatenate(
        (numerical_data, four_categories_encoding, fourty_categories_encoding),
        axis=1,
    )
    y = np.random.randint(1, 7, nrows)
    return X, y


@pytest.fixture(scope="session")
def write_to_file(tmpdir_factory):
    """ "
    Saves data in isolated filesystem.
    """
    X_train, y_train = generate_toy_dataset(40)
    X_test, _ = generate_toy_dataset(42)
    columns = [f"column_{x}" for x in range(0, 55)]
    df_train = pd.DataFrame(X_train, columns=columns)
    df_test = pd.DataFrame(X_test, columns=columns)
    df_train["cover_type"] = y_train

    filename_train = str(tmpdir_factory.mktemp("data").join("train.csv"))
    filename_test = str(tmpdir_factory.mktemp("data").join("test.csv"))

    df_train.to_csv(filename_train)
    df_test.to_csv(filename_test)
    return filename_train, filename_test
