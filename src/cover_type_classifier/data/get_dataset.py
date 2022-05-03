import pandas as pd


def get_dataset(data_path, test_path, nrows=None):
    """
    Gets train and test data from csv file.
    """

    df_train = pd.read_csv(data_path, nrows=nrows)
    df_train.columns = df_train.columns.str.lower()

    X_train = df_train.drop("cover_type", axis=1)
    y_train = df_train["cover_type"]

    X_test = pd.read_csv(test_path, nrows=nrows)
    X_test.columns = X_test.columns.str.lower()

    return X_train, y_train, X_test
