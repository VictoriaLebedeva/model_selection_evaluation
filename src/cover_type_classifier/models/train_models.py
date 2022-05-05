import click
import os
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from cover_type_classifier.data import get_dataset
from cover_type_classifier.data import feature_engineering

# kNN parameter grid

knn_parameters_grid = {
    "n_neighbors": np.arange(1, 20, 1),
    "weights": ["uniform", "distance"],
}

# model parameter grid
random_forest_parameters_grid = {
    "max_features": ["auto", "sqrt", "log2"],
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
        default="models/",
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
        default=True,
        type=bool,
        show_default=True,
        help="Use MinMaxScaler in data preprocessing.",
    )(function)
    function = click.option(
        "--remove-irrelevant-features",
        default=True,
        type=bool,
        show_default=True,
        help="Dimetion reduction by removing irrelevant features.",
    )(function)
    function = click.option(
        "--auto-param-tuning",
        default=False,
        type=bool,
        show_default=True,
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
    train(
        dataset_path,
        test_path,
        prediction_path,
        nrows,
        min_max_scaler,
        remove_irrelevant_features,
        auto_param_tuning,
        model_name,
        model_parameters,
    )


@click.command()
@common_options
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
def random_forest_train(
    dataset_path: str,
    test_path: str,
    prediction_path: str,
    nrows: int,
    max_features: str,  # check this
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
    train(
        dataset_path,
        test_path,
        prediction_path,
        nrows,
        min_max_scaler,
        remove_irrelevant_features,
        auto_param_tuning,
        model_name,
        model_parameters,
    )


def train(
    dataset_path: str,
    test_path: str,
    prediction_path: str,
    nrows: int,
    min_max_scaler: bool,
    remove_irrelevant_features: bool,
    auto_param_tuning: bool,
    model_name: str,
    model_parameters: dict,
) -> None:

    # get_data
    X_train, y_train, X_test = get_dataset.get_dataset(
        dataset_path, test_path, nrows
    )

    # frame to save predictions
    df = pd.DataFrame(X_test.index, columns=["Id"])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if remove_irrelevant_features:
        X_train, X_test = feature_engineering.remove_irrelevant_features(
            X_train, y_train, X_test
        )

    if min_max_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    # train model and make a prediction
    with mlflow.start_run():
        if model_name == "knn":
            model = KNeighborsClassifier(**model_parameters)
            model_parameters_grid = knn_parameters_grid
        elif model_name == "random_forest":
            model = RandomForestClassifier(**model_parameters)
            model_parameters_grid = random_forest_parameters_grid

        print("Estimator", model)
        model.fit(X_train, y_train)

        # cross-validation
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
            print("Best estimator", model.best_estimator_)

        metrics = ["balanced_accuracy", "f1_weighted", "roc_auc_ovo"]

        print("Cross-Validation score results")
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
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

        mlflow.log_param(
            "remove_irrelevant_features", remove_irrelevant_features
        )
        mlflow.log_param("min_max_scaler", min_max_scaler)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("f1_weighted", metrics_scores["f1_weighted"])

        if not auto_param_tuning:
            for param_name, param_value in model_parameters.items():
                mlflow.log_param(param_name, param_value)

        else:
            for param_name, param_value in model.best_params_.items():
                mlflow.log_param(param_name, param_value)

    y_pred = model.predict(X_test)
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
