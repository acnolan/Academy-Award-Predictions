import pandas as pd
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Impport some classifiers to try out
from sklearn.naive_bayes import GaussianNB

# Check out: https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
def preprocessData(trainData, testData):
    # Ignore rows with missing data/NaNs in training
    # In this dataset, that is mostly rows containing data for categories we are not interested in predicting so it should be ok
    trainData.dropna(inplace = True)

    # For test data, we want to predict everything, so we'll need to impute missing values
    imputer = SimpleImputer(missing_values=nan, strategy='mean')
    imputer = imputer.fit(testData[['rating']])
    testData['rating'] = imputer.transform(testData[['rating']])
    
    # Standardization of numeric columns
    # Do we need this?
    featuresToScale = ['rating','sentiment','subjectivity','average_likes','average_retweets']
    std = StandardScaler()
    trainData[featuresToScale] = std.fit_transform(trainData[featuresToScale])
    testData[featuresToScale] = std.transform(testData[featuresToScale])

    # Convert categoricals to numeric values
    # Using 1-hot encoding
    categoricalFeatures = ['category','name','film','genre']

    return trainData

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
    trainData = pd.read_csv("./train.csv")
    testData = pd.read_csv("./test.csv")
    df = preprocessData(trainData, testData)
    #naiveBayes(df)
    return