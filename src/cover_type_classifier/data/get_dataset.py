import click
import pandas as pd
from sklearn.model_selection import train_test_split


# def get_dataset(data_path, random_state, test_size):
#     """
#     Gets data from csv file and splits into train and
#     validation sets.
#     """
#     df = pd.read_csv(data_path)
#     df.columns = df.columns.str.lower()

#     X = df.drop('cover_type', axis=1)
#     y = df['cover_type']
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )
#     return X_train, X_val, y_train, y_val

def get_dataset(data_path, test_path, nrows=None):
    """
    Gets train and test data from csv file.
    """

    df_train = pd.read_csv(data_path, nrows=nrows)
    df_train.columns = df_train.columns.str.lower()

    X_train = df_train.drop('cover_type', axis=1)
    y_train = df_train['cover_type']

    X_test = pd.read_csv(test_path, nrows=nrows)
    X_test.columns = X_test.columns.str.lower()

    return X_train, y_train, X_test