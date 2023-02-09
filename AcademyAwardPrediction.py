import argparse
import pandas as pd
from Letterboxd import getAllLetterboxdRatings
from Twitter import getAllTwitterData
#import sklearn

# Set up command line argument parsing and -h/--help flags
parser = argparse.ArgumentParser(
    prog = 'AcademyAwardPrediction',
    description = 'Runs some machine learning on historical Academy Award data as well as Letterboxd and Twitter info on the films.')

# Set up flags for rebuilding the csv with letterboxd and twitter data
parser.add_argument('-b', '--build', action='store_true')

# Set Pandas options
pd.set_option('display.max_columns', None)

# Build the data table
def rebuildTable():
    df = pd.read_csv("./oscar_nominees.csv")

    df2 = df.head(5)

    filmList = df2['film'].to_list()
    yearList = df2['year_film'].to_list()
    categoryList = df2['category'].to_list()

    df2 = pd.concat([df2, getAllLetterboxdRatings(filmList,yearList)], axis=1)
    df2 = pd.concat([df2, getAllTwitterData(filmList)], axis=1)

    print(df2)
    df2.to_csv('oscar_nominees_full_columns.csv')

def engageMachineLearningAlgorithms():
    print("Todo: Machine learning!")


# Optional -b flag for building
if __name__ == "__main__":
    # If the build flag is present, rebuild the data
    # Building is slow so this is optional
    args = parser.parse_args()
    if args.build:
        rebuildTable()

    # Run the fancy machine learning?
    engageMachineLearningAlgorithms()