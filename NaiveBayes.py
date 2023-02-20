import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from VisualizeData import plotConfusionMatrix

# Train a Gaussian Naive Bayes model
def trainNaiveBayes(df):
    print('Starting Naive Bayes')

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

    print("Training Accuracy for Naive Bayes {:.4}%".format(accuracy))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    plotConfusionMatrix(y_test, y_pred)

    return gnb, accuracy

# Run the Naive Bayes model on the unknown data
# To avoid multiples in a same category winning, we'll also record their probabilities
# We can call the winner the higher probability
def testNaiveBayes(gnb, df, original):
    predictionDictionary = {}
    predictionDictionary['category'] = original['category']
    predictionDictionary['film'] = original['film']

    df = df.drop(['winner'], axis=1)

    predictionDictionary['predictions'] = gnb.predict(df.values)
    predictionProbabity = gnb.predict_proba(df.values)

    predictionDictionary['probability_loses'] = [p[0] for p in predictionProbabity]
    predictionDictionary['probability_wins'] = [p[1] for p in predictionProbabity]
    
    predicted_df = pd.DataFrame(predictionDictionary)

    predicted_df.to_csv('NaiveBayes_predictions.csv')

    return predicted_df