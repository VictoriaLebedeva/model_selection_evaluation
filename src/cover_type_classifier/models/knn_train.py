import click 
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
    '-p',
    '--prediction-path',
    default='models/',
    type=click.Path(exists=True, dir_okay=True),
    show_default=True,
)

@click.option(
    '--random-state',
    default=42,
    type=int,
    show_default=True,
)

@click.option(
    '--test-size',
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
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
    

def train(dataset_path, prediction_path, random_state, test_size, n_neighbors, weights):

    # X_train, X_val, y_train, y_val = get_dataset.get_dataset(dataset_path, random_state, test_size) # return

    
    # to be removed
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.lower()

    X = df.drop('cover_type', axis=1)
    y = df['cover_type']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # to be removed 


    # train model and make a prediction
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    print('Estimator', knn)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    # generate name of the output file
    now = datetime.now()
    report_filename = f'prediction_knn_{now.strftime("%d%m%Y_%H%M%S")}.csv'
    output_path = os.path.join(prediction_path, report_filename)

    # save prediction to csv
    df = pd.DataFrame(X_val.index, columns=['Id'])
    df['Cover_Type'] = y_pred
    df.to_csv(output_path, index=False)  

if __name__ == '__main__':
    train()