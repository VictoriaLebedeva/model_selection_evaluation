import click
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
# from cover_type_classifier.data import get_dataset  # return
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
    '--max-features',
    default='auto',
    show_default=True,
    help="Maximum features used for each tree."
)
@click.option(
    '--n-estimators',
    default=50,
    type=click.IntRange(1),
    show_default=True,
    help="The number of trees in the forest."
)
@click.option(
    '--min-samples-leaf',
    default=50,
    type=click.IntRange(1),
    show_default=True,
    help="The minimum number of samples required to be at a leaf node."
)
def train(dataset_path,
          test_path,
          prediction_path,
          nrows,
          max_features,
          n_estimators,
          min_samples_leaf):

    # X_train, X_val, y_train, y_val = get_dataset.get_dataset(dataset_path, random_state, test_size) # return
    # X_train, X_val, y_train, y_val = get_dataset.get_dataset(dataset_path, test_path, nrows) # return

    # to be removed
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv(dataset_path, nrows=200)
    df_train.columns = df_train.columns.str.lower()

    X_train = df_train.drop('cover_type', axis=1)
    y_train = df_train['cover_type']

    X_test = pd.read_csv(test_path, nrows=100)
    X_test.columns = X_test.columns.str.lower()

    # to be removed

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)

    # train model and make a prediction
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, )
    print('Estimator', rf_clf)    
    rf_clf.fit(X_train, y_train)

    # cross-validation
    metrics = ['balanced_accuracy', 'f1_weighted', 'roc_auc_ovo']
    print('Cross-Validation score results')
    for metric in metrics:
        print(f'{metric}:', cross_val_score(rf_clf, X_train_shuffled, y_train_shuffled, cv=5, scoring=metric))

    y_pred = rf_clf.predict(X_test)

    # generate name of the output file
    now = datetime.now()
    report_filename = f'prediction_random_forest_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df = pd.DataFrame(X_test.index, columns=['Id'])
    df['Cover_Type'] = y_pred
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    train()
