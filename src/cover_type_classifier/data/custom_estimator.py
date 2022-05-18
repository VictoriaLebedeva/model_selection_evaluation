import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif


class RemoveIrrelevantFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.chi2_selector = SelectPercentile(chi2, percentile=75)
        self.fvalue_selector = SelectPercentile(f_classif, percentile=75)

    def fit(self, X, y):
        # split features into features

        if isinstance(X, pd.DataFrame):
            X = X.copy().values

        self.numerical_features = X[:, 1:11]
        self.categorical_features = X[:, 11:]

        self.chi2_selector.fit(self.categorical_features, y)
        self.fvalue_selector.fit(self.numerical_features, y)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy().values        
        numerical_features = X[:, 1:11]
        categorical_features = X[:, 11:]

        categorical_features = self.chi2_selector.transform(
            categorical_features)
        numerical_features = self.fvalue_selector.transform(numerical_features)
        X_processed = np.concatenate(
            (numerical_features, categorical_features), axis=1
        )
        return X_processed


class NoTransformation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X
