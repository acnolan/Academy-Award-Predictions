import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from VisualizeData import plotConfusionMatrix

KNNParamGrid = {
    'n_neighbors': [1, 3, 5, 10, 15, 20, 25, 50],
    'p': [1, 2]
}

# Train a K-Nearest Neighbors model
def trainKNN(df):
    print('Trying KNN...')

    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Tune hyperparameters and train the model
    knn = tuneHyperparameters(X_train, y_train)
    
    # Test the model
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print("Training Accuracy for KNN {:.4}%".format(accuracy * 100))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)
    return knn, accuracy

# Test various hyperparameters to see what works best
def tuneHyperparameters(X, y):
    randomSearch = GridSearchCV(KNeighborsClassifier(), KNNParamGrid)
    randomSearch.fit(X, y)

    print("Best hyperparameters for KNN: ", randomSearch.best_estimator_)

    return randomSearch.best_estimator_
