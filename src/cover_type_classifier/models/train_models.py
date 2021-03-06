import click
import os
import joblib
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.cover_type_classifier.data import get_dataset
from src.cover_type_classifier.data import feature_engineering

# kNN parameter grid
knn_parameters_grid = {
    "n_neighbors": np.arange(1, 20, 1),
    "weights": ["uniform", "distance"],
}


# model parameter grid
random_forest_parameters_grid = {
    "max_features": ["sqrt", "log2"],
    "n_estimators": np.arange(10, 50, 10),
    "min_samples_leaf": np.arange(50, 300, 50),
}


# common model options
def common_options(function):
    function = click.option(
        "-d",
        "--dataset-path",
        default="data/external/train.csv",
        type=click.Path(exists=True, dir_okay=False),
        show_default=True,
    )(function)
    function = click.option(
        "-t",
        "--test-path",
        default="data/external/test.csv",
        type=click.Path(exists=True, dir_okay=False),
        show_default=True,
    )(function)
    function = click.option(
        "-p",
        "--prediction-path",
        default="models/predictions/",
        type=click.Path(exists=True, dir_okay=True),
        show_default=True,
    )(function)
    function = click.option(
        "--nrows",
        default=None,
        type=click.IntRange(1),
        show_default=True,
    )(function)
    function = click.option(
        "--min-max-scaler",
        is_flag=True,
        help="Use MinMaxScaler in data preprocessing.",
    )(function)
    function = click.option(
        "--remove-irrelevant-features",
        is_flag=True,
        help="Dimetion reduction by removing irrelevant features.",
    )(function)
    function = click.option(
        "--auto-param-tuning",
        is_flag=True,
        help="Use automated parameter tuning.",
    )(function)
    return function


@click.command()
@common_options
@click.option(
    "--n-neighbors",
    default=5,
    type=click.IntRange(1),
    show_default=True,
    help="Number of neighbors",
)
@click.option(
    "-w",
    "--weights",
    default="uniform",
    type=click.Choice(["uniform", "distance"]),
    show_default=True,
    help="kNN model weights.",
)
def knn_train(
    dataset_path: str,
    test_path: str,
    prediction_path: str,
    nrows: int,
    n_neighbors: int,
    weights: str,
    min_max_scaler: bool,
    remove_irrelevant_features: bool,
    auto_param_tuning: bool,
) -> None:

    model_name = "knn"
    model_parameters = {"n_neighbors": n_neighbors, "weights": weights}
    feature_engineering_parameters = {
        "min_max_scaler": min_max_scaler,
        "remove_irrelevant_features": remove_irrelevant_features,
    }

    # get data
    X_train, y_train, X_test = get_dataset.get_dataset(
        dataset_path, test_path, nrows
    )
    index = X_test.index
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # process data
    pipeline = feature_engineering.transformation_pipeline(
        remove_irrelevant_features, min_max_scaler
    )
    pipeline.fit(X_train, y_train)
    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)

    y_pred, estimator = train(
        X_train,
        y_train,
        X_test,
        feature_engineering_parameters,
        auto_param_tuning,
        model_name,
        model_parameters,
    )

    print("Do you want to save model prediction? (y/n)")
    is_save_model_prediction = input()
    print("Do you want to save model? (y/n)")
    is_save_model = input()

    if is_save_model_prediction.lower().strip() == "y":
        write_prediction_to_file(index, y_pred, model_name, prediction_path)
    if is_save_model.lower().strip() == "y":
        save_model(estimator, pipeline, model_name)


@click.command()
@common_options
@click.option(
    "--max-features",
    default="sqrt",
    type=click.Choice(["sqrt", "log2"]),
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
def random_forest_train(
    dataset_path: str,
    test_path: str,
    prediction_path: str,
    nrows: int,
    max_features: str,
    n_estimators: int,
    min_samples_leaf: int,
    min_max_scaler: bool,
    remove_irrelevant_features: bool,
    auto_param_tuning: bool,
) -> None:

    model_name = "random_forest"
    model_parameters = {
        "max_features": max_features,
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
    }

    feature_engineering_parameters = {
        "min_max_scaler": min_max_scaler,
        "remove_irrelevant_features": remove_irrelevant_features,
    }
    # get data
    X_train, y_train, X_test = get_dataset.get_dataset(
        dataset_path, test_path, nrows
    )
    index = X_test.index
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # process data
    pipeline = feature_engineering.transformation_pipeline(
        remove_irrelevant_features, min_max_scaler
    )
    pipeline.fit(X_train, y_train)
    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)

    y_pred, estimator = train(
        X_train,
        y_train,
        X_test,
        feature_engineering_parameters,
        auto_param_tuning,
        model_name,
        model_parameters,
    )

    print("Do you want to save model prediction? (y/n)")
    is_save_model_prediction = input()
    print("Do you want to save model? (y/n)")
    is_save_model = input()

    if is_save_model_prediction.lower().strip() == "y":
        write_prediction_to_file(index, y_pred, model_name, prediction_path)
    if is_save_model.lower().strip() == "y":
        save_model(estimator, pipeline, model_name)


def write_prediction_to_file(
    index: pd.Series, y_pred: pd.Series, model_name: str, prediction_path: str
) -> str:
    df = pd.DataFrame(index, columns=["Id"])

    # generate name of the output file
    now = datetime.now()
    report_filename = (
        f'prediction_{model_name}_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    )
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df["Cover_Type"] = y_pred
    df.to_csv(output_path, index=False)
    print(f"Model output was saved to {output_path}")
    return output_path


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    feature_engineering_parameters: dict,
    auto_param_tuning: bool,
    model_name: str,
    model_parameters: dict,
) -> pd.Series:

    # train model and make a prediction
    with mlflow.start_run():
        if model_name == "knn":
            model = KNeighborsClassifier(**model_parameters)
            model_parameters_grid = knn_parameters_grid
        elif model_name == "random_forest":
            model = RandomForestClassifier(**model_parameters)
            model_parameters_grid = random_forest_parameters_grid

        # nested cross-validation
        if auto_param_tuning:
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            model = GridSearchCV(
                model,
                model_parameters_grid,
                scoring="f1_weighted",
                n_jobs=1,
                cv=cv_inner,
                refit=True,
            )

        model.fit(X_train, y_train)
        print("Estimator", model)

        metrics = ["balanced_accuracy", "f1_weighted", "roc_auc_ovo"]
        print("Cross-Validation score results")
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics_scores = {}

        for metric in metrics:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring=metric,
                cv=cv_outer,
                n_jobs=None,
            )
            metrics_scores[metric] = np.mean(scores)
            print(f"{metric}:", scores)

        for param_name, param_value in feature_engineering_parameters.items():
            mlflow.log_param(param_name, param_value)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("f1_weighted", metrics_scores["f1_weighted"])

        if not auto_param_tuning:
            for param_name, param_value in model_parameters.items():
                mlflow.log_param(param_name, param_value)

        else:
            for param_name, param_value in model.best_params_.items():
                mlflow.log_param(param_name, param_value)

    y_pred = model.predict(X_test)
    return y_pred, model


def save_model(estimator: object, pipeline: Pipeline, model_name: str) -> None:
    """Saves fitted model into .bin file"""
    now = datetime.now()
    model_filename = f'{model_name}_{now.strftime("%d%m%Y_%H%M%S")}.bin'
    path = os.path.join("models/models", model_filename)
    with open(path, "wb") as f_out:
        joblib.dump((estimator, pipeline), f_out)
        print(f"Model was succesfully saved in {path}!")
