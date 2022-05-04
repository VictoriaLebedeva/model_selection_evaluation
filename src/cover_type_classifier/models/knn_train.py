import click
import os
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from cover_type_classifier.data import get_dataset
from cover_type_classifier.data import feature_engineering

# mlflow experiment identifier
mlflow_experiment_id = 1

# model parameter grid
param = {
    'n_neighbors': np.array(1, 20, 1),
    'weights': ["uniform", "distance"]
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
    n_neighbors: int,
    weights: str,
    min_max_scaler: bool,
    remove_irrelevant_features: bool,
) -> None:

    # get data
    X_train, y_train, X_test = get_dataset.get_dataset(
        dataset_path, test_path, nrows
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if remove_irrelevant_features:
        X_train = feature_engineering.remove_irrelevant_features(X_train, y_train)
    
    if min_max_scaler:
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = scaler.fit_transform(X_train)


    # train model and make a prediction
    with mlflow.start_run(experiment_id=mlflow_experiment_id):
        
        # model initialization
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        print("Estimator", knn)
        knn.fit(X_train, y_train)

        # cross-validation
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        search = GridSearchCV(knn, param, scoring='f1_weighted', n_jobs=1, cv=cv_inner, refit=True)
        metrics = ["balanced_accuracy", "f1_weighted", "roc_auc_ovo"]
        print("Cross-Validation score results")

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        metrics_scores = {} 

        for metric in metrics:
            scores = cross_val_score(search, X_train, y_train, scoring='f1_weighted', cv=cv_outer, n_jobs=-1)
            metrics_scores[metric] = np.mean(scores)
            print(f"{metric}:", scores)

        mlflow.log_param('n_neighbors', n_neighbors)
        mlflow.log_param('weights', weights)
        mlflow.log_metric('f1_weighted', metrics_scores['f1_weighted'])
        mlflow.sklearn.log_model(knn, 'model')

    y_pred = knn.predict(X_test)

    # generate name of the output file
    now = datetime.now()
    report_filename = f'prediction_knn_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df = pd.DataFrame(X_test.index, columns=["Id"])
    df["Cover_Type"] = y_pred
    df.to_csv(output_path, index=False)
    print(f"Model output was saved to {output_path}")


if __name__ == "__main__":
    train()
