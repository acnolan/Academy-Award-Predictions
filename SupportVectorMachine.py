import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from VisualizeData import plotConfusionMatrix

# Train a Support Vector Machine model
def trainSVM(df):
    print('Trying Support Vector Machine...')

    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Train the model
    svc = SVC(probability = True)
    svc.fit(X_train, y_train)
    
    # Test the model
    y_pred = svc.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print("Training Accuracy for Support Vector Machine {:.4}%".format(accuracy * 100))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)
    return svc, accuracy
