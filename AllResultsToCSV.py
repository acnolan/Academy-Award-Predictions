# A quick script to get the winners from the CSV files and put them all into a big spreadsheet
import pandas as pd
from MachineLearning import determineWinnerFromPredictions

# Load data
naiveBayes = pd.read_csv("./NaiveBayes_predictions.csv")
Knn = pd.read_csv("./KNN_predictions.csv")
LogisticRegression = pd.read_csv("./LogisticRegression_predictions.csv")
RandomForest = pd.read_csv("./RandomForest_predictions.csv")
Svm = pd.read_csv("./SVM_predictions.csv")

# Determine winners
naiveBayesWinners = determineWinnerFromPredictions(naiveBayes)
KnnWinners = determineWinnerFromPredictions(Knn)
LogisticRegressionWinners = determineWinnerFromPredictions(LogisticRegression)
RandomForestWinners = determineWinnerFromPredictions(RandomForest)
SvmWinners = determineWinnerFromPredictions(Svm)

#
def getWinnerNamesInOrder(categories, winnerDict):
    l = []
    for c in categories:
        l.append(winnerDict[c]['winner'])
    return l

# Get the winner values
categories = naiveBayesWinners.keys()
naiveBayesList = getWinnerNamesInOrder(categories, naiveBayesWinners)
KnnList = getWinnerNamesInOrder(categories, KnnWinners)
LogisticRegressionList = getWinnerNamesInOrder(categories, LogisticRegressionWinners)
RandomForestList = getWinnerNamesInOrder(categories, RandomForestWinners)
SvmList = getWinnerNamesInOrder(categories, SvmWinners)

# Build a big ol dataframe
table = pd.DataFrame(list(zip(naiveBayesList, KnnList, LogisticRegressionList, RandomForestList, SvmList)), index=categories, columns=['Naive Bayes','KNN','Logistic Regression','Random Forest','SVM'])

table.to_csv('BigResultsSpreadsheet.csv')