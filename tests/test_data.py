# import pytest
import numpy as np
import pandas as pd


def generate_toy_dataset() -> np.ndarray:
    nrows = 1000
    np.random.seed(42)

    fourty_categories_encoding = pd.get_dummies(
        pd.Series(np.random.randint(0, 40, nrows))
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


if __name__ == "__main__":
    X, y = generate_toy_dataset()


def test_create_file(tmpdir):
    p = tmpdir.mkdir("sub").join("hello.txt")
    p.write("content")
    assert p.read() == "content"
    assert len(tmpdir.listdir()) == 1
