import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from VisualizeData import plotConfusionMatrix

RandomForestParamGrid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9, None],
    'max_leaf_nodes': [3, 6, 9, None],
    'min_samples_split': [2, 4, 8]
}

# Train a Random Forest model
def trainRandomForest(df):
    print('Trying Random Forest...')

    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Tune hyperparameters and train the model
    rf = tuneHyperparameters(X_train, y_train)
    
    # Test the model
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print("Training Accuracy for Random Forest {:.4}%".format(accuracy * 100))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)
    return rf, accuracy

# Test various hyperparameters to see what works best
def tuneHyperparameters(X, y):
    randomSearch = RandomizedSearchCV(RandomForestClassifier(), RandomForestParamGrid)
    randomSearch.fit(X, y)

    print("Best hyperparameters for Random Forest: ", randomSearch.best_estimator_)

    return randomSearch.best_estimator_
