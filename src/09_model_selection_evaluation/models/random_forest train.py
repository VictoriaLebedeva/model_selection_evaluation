import click 
from sklearn.neighbors import KNeighborsClassifier


def train():
    knn = KNeighborsClassifier()
    knn.fit(X_train)
    knn.predict(X_test)

if __name__ == '__main__':
    train()