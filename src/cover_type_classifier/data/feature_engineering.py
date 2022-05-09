import pandas as pd
import numpy as np

from typing import Tuple

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif


def remove_irrelevant_features(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits feature set into categorical and numeric and
    return new feature set with the most relevant features.
    """

    # split features
    numerical_features = X_train.iloc[:, 1:11]
    categorical_features = X_train.iloc[:, 11:]

    numerical_features_test = X_test.iloc[:, 1:11]
    categorical_features_test = X_test.iloc[:, 11:]

    # select best features among categorical
    chi2_selector = SelectPercentile(chi2, percentile=75)
    chi2_selector.fit(categorical_features, y_train)
    categorical_features = chi2_selector.transform(categorical_features)

    categorical_features_test = chi2_selector.transform(
        categorical_features_test
    )

    # select best features among numerical
    fvalue_selector = SelectPercentile(f_classif, percentile=75)
    fvalue_selector.fit(numerical_features, y_train)
    numerical_features = fvalue_selector.transform(numerical_features)
    numerical_features_test = fvalue_selector.transform(
        numerical_features_test
    )

    X_train_processed = np.concatenate(
        (numerical_features, categorical_features), axis=1
    )
    X_test_processed = np.concatenate(
        (numerical_features_test, categorical_features_test), axis=1
    )

    print("Initial number of features (train):", X_train.shape[1])
    print("Reduced number of features (train):", X_train_processed.shape[1])

    print("Initial number of features (test):", X_test.shape[1])
    print("Reduced number of features (test):", X_test_processed.shape[1])

    return X_train_processed, X_test_processed


def process_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    remove_irrelevant_features_flag: bool,
    min_max_scaler: bool,
) -> Tuple[np.ndarray, pd.Series, np.ndarray]:

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if remove_irrelevant_features_flag:
        X_train, X_test = remove_irrelevant_features(X_train, y_train, X_test)

    if min_max_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    return X_train, y_train, X_test


if __name__ == "__main__":
    df = pd.read_csv("data/external/train.csv")
    target = df["Cover_Type"]
    features = df.drop("Cover_Type", axis=1)
    features_test = pd.read_csv("data/external/test.csv")
    remove_irrelevant_features(features, target, features_test)
