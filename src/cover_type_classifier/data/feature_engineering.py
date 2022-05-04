import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif


def remove_irrelevant_features(
    X_train: pd.DataFrame, y_train: pd.Series
) -> np.ndarray:
    """Splits feature set into categorical and numeric and
    return new feature set with the most relevant features.
    """

    # split features
    numerical_features = X_train.iloc[:, 1:11]
    categorical_features = X_train.iloc[:, 11:]

    # select best features among categorical
    chi2_selector = SelectPercentile(chi2, percentile=75)
    categorical_features = chi2_selector.fit_transform(
        categorical_features, y_train
    )

    # select best features among numerical
    fvalue_selector = SelectPercentile(f_classif, percentile=75)
    numerical_features = fvalue_selector.fit_transform(
        numerical_features, y_train
    )

    X_train_processed = np.concatenate(
        (numerical_features, categorical_features), axis=1
    )

    print("Initial number of features:", X_train.shape[1])
    print("Reduced number of features:", X_train_processed.shape[1])

    return X_train_processed


if __name__ == "__main__":
    df = pd.read_csv("data\\external\\train.csv")
    target = df["Cover_Type"]
    features = df.drop("Cover_Type", axis=1)
    remove_irrelevant_features(features, target)
