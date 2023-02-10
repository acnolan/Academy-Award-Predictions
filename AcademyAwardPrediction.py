import argparse
import pandas as pd
from Letterboxd import getLetterboxdMovieDetails
from Twitter import getTwitterData
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

    ratingList = []
    genreList = []
    sentiments = []
    subjectivity = []
    likes = []
    retweets = []

    for i, film in enumerate(filmList):
        letterBoxdData = getLetterboxdMovieDetails(film, yearList[i])
        twitterData = getTwitterData(film)
        ratingList.append(letterBoxdData['rating'])
        genreList.append(letterBoxdData['genre'])
        sentiments.append(twitterData['sentiment'])
        subjectivity.append(twitterData['subjectivity'])
        likes.append(twitterData['likeCount'])
        retweets.append(twitterData['retweetCount'])

    df2['rating'] = ratingList
    df2['genre'] = genreList
    df2['sentiment'] = sentiments
    df2['subjectivity'] = subjectivity
    df2['average likes'] = likes
    df2['average retweets'] = retweets

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