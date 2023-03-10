import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from VisualizeData import plotConfusionMatrix

# Train a Gaussian Naive Bayes model
def trainNaiveBayes(df):
    print('Trying Naive Bayes...')

    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Fit the model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Test the model
    y_pred = gnb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print("Training Accuracy for Naive Bayes {:.4}%".format(accuracy * 100))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)

    return gnb, accuracy
