import click
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from cover_type_classifier.data import get_dataset
from datetime import datetime


@click.command()
@click.option(
    '-d',
    '--dataset-path',
    default='data/external/train.csv',
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    '-t',
    '--test-path',
    default='data/external/test.csv',
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    '-p',
    '--prediction-path',
    default='models/',
    type=click.Path(exists=True, dir_okay=True),
    show_default=True,
)
@click.option(
    '--nrows',
    default=None,
    type=click.IntRange(1),
    show_default=True,
)
@click.option(
    '--n-neighbors',
    default=5,
    type=click.IntRange(1),
    show_default=True,
    help="Number of neighbors"
)
@click.option(
    '-w',
    '--weights',
    default='uniform',
    type=click.Choice(['uniform', 'distance']),
    show_default=True,
    help="kNN model weights."
)
def train(dataset_path,
          test_path,
          prediction_path,
          nrows,
          n_neighbors,
          weights):

    # get data 
    X_train, y_train, X_test = get_dataset.get_dataset(dataset_path, test_path, nrows)
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)

    # train model and make a prediction
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    print('Estimator', knn)    
    knn.fit(X_train, y_train)

    # cross-validation
    metrics = ['balanced_accuracy', 'f1_weighted', 'roc_auc_ovo']
    print('Cross-Validation score results')
    for metric in metrics:
        print(f'{metric}:', cross_val_score(knn, X_train_shuffled, y_train_shuffled, cv=5, scoring=metric))

    y_pred = knn.predict(X_test)

    # generate name of the output file
    now = datetime.now()
    report_filename = f'prediction_knn_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df = pd.DataFrame(X_test.index, columns=['Id'])
    df['Cover_Type'] = y_pred
    df.to_csv(output_path, index=False)
    print(f'Model output was saved to {output_path}')


if __name__ == '__main__':
    train()
