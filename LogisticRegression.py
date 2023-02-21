import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from VisualizeData import plotConfusionMatrix

LogisticRegressionParamGrid = {
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

# Train a Logistic Regression model
def trainLogisticRegression(df):
    print('Trying Logistic Regression...')

    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Tune hyperparameters and train the model
    lr = tuneHyperparameters(X_train, y_train)
    
    # Test the model
    y_pred = lr.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print("Training Accuracy for Logistic Regression {:.4}%".format(accuracy * 100))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)
    return lr, accuracy

# Test various hyperparameters to see what works best
def tuneHyperparameters(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        randomSearch = GridSearchCV(LogisticRegression(), LogisticRegressionParamGrid)

        randomSearch.fit(X, y)

    print("Best hyperparameters for Logistic Regression: ", randomSearch.best_estimator_)

    return randomSearch.best_estimator_