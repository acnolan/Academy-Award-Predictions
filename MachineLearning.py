import pandas as pd
from numpy import nan
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from NaiveBayes import trainNaiveBayes, testNaiveBayes
from VisualizeData import plotCorrelationHeatmap

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

    # Make the 'category' column lower case to be more standardized
    trainData['category'] = trainData['category'].str.lower()
    testData['category'] = testData['category'].str.lower()

    trainData = pd.get_dummies(trainData, columns=categoricalFeatures, drop_first=True)
    testData = pd.get_dummies(testData, columns=categoricalFeatures, drop_first=True)

    # Align data so the columns are ok after one hot, this may lose some data as categories have changed over time
    # But for now this is a safe way to handle it since we will only be predicting the modern movies
    trainData, testData = trainData.align(testData, join='inner', axis=1)

    # Drop the columns we won't need
    # We already frequency mapped them when building the dataset
    dropableColumns = ['year_film','year_ceremony','ceremony','name','film']
    trainData.drop(dropableColumns, axis=1, inplace=True)
    testData.drop(dropableColumns, axis=1, inplace=True)

    plotCorrelationHeatmap(trainData)

    return trainData, testData

# Check which movie has the highest win probability
# That's our winner!
def determineWinnerFromPredictions(df):
    winnerDict = {}

    for _, row in df.iterrows():
        # If we see a higher probability in the category, that's our new winner
        if row['category'] in winnerDict:
            if row['probability_wins'] > winnerDict[row['category']]['prob']:
                d = {}
                d['winner'] = row['film']
                d['prob'] = row['probability_wins']
                winnerDict[row['category']] = d
        else:
            d = {}
            d['winner'] = row['film']
            d['prob'] = row['probability_wins']
            winnerDict[row['category']] = d

    return winnerDict

# Print winners
def printWinner(winnerDict):
    for category, value in winnerDict.items():
        print(category, " winner is ", value['winner'], "!")
    return

# Train and run the machine learning algorithms
def executeMachineLearning():
    # Load original raw data
    trainOriginal = pd.read_csv("./train.csv")
    testOriginal = pd.read_csv("./test.csv")

    # Preprocess the data
    trainData, testData = preprocessData(trainOriginal, testOriginal)
    
    # Try out naive bayes
    gnb, gnb_accuracy = trainNaiveBayes(trainData)
    predicted_gnb = testNaiveBayes(gnb, testData, testOriginal)

    # Output the winners
    printWinner(determineWinnerFromPredictions(predicted_gnb))
    return