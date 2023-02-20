import pandas as pd
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Impport some classifiers to try out
from sklearn.naive_bayes import GaussianNB

# Run some basic preprocessing to remove null values, standardize numbers, and convert nominal variables to one hot variables
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

    # Convert nominal categoricals to numeric values
    # Using 1-hot encoding for genre and category, use drop_first to avoid multicollinearity
    # Film and nominee name are already handled by frequency encoding
    categoricalFeatures = ['category','genre']

    trainData = pd.get_dummies(trainData, columns=categoricalFeatures, drop_first=True)
    testData = pd.get_dummies(testData, columns=categoricalFeatures, drop_first=True)

    # Align data so the columns are ok after one hot, this may lose some data as categories have changed over time
    # But for now this is a safe way to handle it since we will only be predicting the modern movies
    trainData, testData = trainData.align(testData, join='inner', axis=1)

    # Drop the columns we won't need
    # We already frequency mapped them when building the dataset
    dropableColumns = ['name','film']
    trainData.drop(dropableColumns, axis=1, inplace=True)
    testData.drop(dropableColumns, axis=1, inplace=True)

    return trainData, testData

# We won't do much with this yet, but could be good to see
# Also good practice
def visualizeData():
    # Maybe make a separate file for this
    return


# Try out: https://www.youtube.com/watch?v=99MN-rl8jGY&ab_channel=TEW22
def trainNaiveBayes(df):
    X = df.drop(['winner'],axis=1).values

    # Target column
    y = df['winner'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Fit the model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Test the model
    y_pred = gnb.predict(X_test)

    print("Training Accuracy {:.4}%".format(accuracy_score(y_test, y_pred)))
    print("Number of mislabeled training awards out of a total %d entries: %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    return gnb

# Run the Naive Bayes model on the unknown data
# To avoid multiples in a same category winning, we'll also record their probabilities
# We can call the winner the higher probability
def testNaiveBayes(gnb, df, original):
    df = df.drop(['winner'], axis=1)
    predictions = gnb.predict(df)
    predictionProbabity = gnb.predict_proba(df)

    # TODO: maybe something better than printing
    for i, p in enumerate(predictions):
        print("Category: ", original['category'][i], "Movie: ", original['film'][i], ", wins: ", p, ", probability: ", predictionProbabity[i])
    
    return

def executeMachineLearning():
    trainOriginal = pd.read_csv("./train.csv")
    testOriginal = pd.read_csv("./test.csv")
    trainData, testData = preprocessData(trainOriginal, testOriginal)
    gnb = trainNaiveBayes(trainData)
    testNaiveBayes(gnb, testData, testOriginal)
    return