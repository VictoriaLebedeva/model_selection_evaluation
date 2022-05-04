import click
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from cover_type_classifier.data import get_dataset
from cover_type_classifier.data import feature_engineering
from datetime import datetime

import mlflow


# model parameter grid
param = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': np.arange(10, 50, 10),
    'min_samples_leaf': np.arange(50, 300, 50),
}


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/external/train.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-t",
    "--test-path",
    default="data/external/test.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-p",
    "--prediction-path",
    default="models/",
    type=click.Path(exists=True, dir_okay=True),
    show_default=True,
)
@click.option(
    "--nrows",
    default=None,
    type=click.IntRange(1),
    show_default=True,
)
@click.option(
    "--max-features",
    default="auto",
    show_default=True,
    help="Maximum features used for each tree.",
)
@click.option(
    "--n-estimators",
    default=50,
    type=click.IntRange(1),
    show_default=True,
    help="The number of trees in the forest.",
)
@click.option(
    "--min-samples-leaf",
    default=50,
    type=click.IntRange(1),
    show_default=True,
    help="The minimum number of samples required to be at a leaf node.",
)
@click.option(
    "--min-max-scaler",
    default=False,
    type=bool,
    show_default=True,
    help="Use MinMaxScaler in data preprocessing.",
)
@click.option(
    "--remove-irrelevant-features",
    default=False,
    type=bool,
    show_default=True,
    help="Dimetion reduction by removing irrelevant features.",
)
def train(
    dataset_path: str,
    test_path: str,
    prediction_path: str,
    nrows: int,
    max_features: str,  # check this
    n_estimators: int,
    min_samples_leaf: int,
    min_max_scaler: bool,
    remove_irrelevant_features: bool,
) -> None:

    X_train, y_train, X_test = get_dataset.get_dataset(
        dataset_path, test_path, nrows
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if remove_irrelevant_features:
        X_train = feature_engineering.remove_irrelevant_features(
            X_train, y_train
        )

    if min_max_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)

    with mlflow.start_run():
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
        )
        print("Estimator", rf_clf)
        rf_clf.fit(X_train, y_train)

        # cross-validation
        # cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # search = GridSearchCV(
        #     rf_clf,
        #     param,
        #     scoring='f1_weighted',
        #     n_jobs=1,
        #     cv=cv_inner,
        #     refit=True,
        # )

        metrics = ["balanced_accuracy", "f1_weighted", "roc_auc_ovo"]
        print("Cross-Validation score results")

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        metrics_scores = {}

        for metric in metrics:
            scores = cross_val_score(
                rf_clf,
                X_train,
                y_train,
                scoring='f1_weighted',
                cv=cv_outer
            )
            metrics_scores[metric] = np.mean(scores)
            print(f"{metric}:", scores)

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_features', max_features)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_param(
            "remove_irrelevant_features", remove_irrelevant_features
        )
        mlflow.log_param("min_max_scaler", min_max_scaler)
        mlflow.log_metric('f1_weighted', metrics_scores['f1_weighted'])
        mlflow.sklearn.log_model(rf_clf, 'model')

    # y_pred = rf_clf.predict(X_test)

    # generate name of the output file
    now = datetime.now()
    report_filename = f'prediction_rf_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df = pd.DataFrame(X_test.index, columns=["Id"])
    # df["Cover_Type"] = y_pred
    df.to_csv(output_path, index=False)

    print(f"Model output was saved to {output_path}")


if __name__ == "__main__":
    train()
