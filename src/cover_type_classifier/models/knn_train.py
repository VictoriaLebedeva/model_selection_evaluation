import click 
from sklearn.neighbors import KNeighborsClassifier
from cover_type_classifier.data import get_dataset 


@click.command()
@click.option(
    '-d',
    '--dataset-path',
    default='data/external/train.csv',
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)

@click.option(
    '-d',
    '--dataset-path',
    default='data/external/train.csv',
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-size",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)

def train(dataset_path, random_state, test_size):

    X_train, X_val, y_train, y_val = get_dataset.get_dataset(dataset_path, random_state, test_size)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn.predict(X_val)

if __name__ == '__main__':
    train()