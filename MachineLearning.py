import pandas as pd
from sklearn.model_selection import train_test_split
# Impport some classifiers to try out
from sklearn.naive_bayes import GaussianNB

# Check out: https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
def preprocessData():
    df = pd.read_csv("./train.csv")

    # Ignore rows with missing data/NaNs
    
    # Standardization of numerics

    # Convert categoricals to numeric values

    return df

# We won't do much with this yet, but could be good to see
# Also good practice
def visualizeData():
    # Maybe make a separate file for this
    return


# Try out: https://www.youtube.com/watch?v=99MN-rl8jGY&ab_channel=TEW22
def naiveBayes(df):
    X = df.drop(['winner'],axis=1).values
    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return
    return

def executeMachineLearning():
    df = preprocessData()
    naiveBayes(df)
    return