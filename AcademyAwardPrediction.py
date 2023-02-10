import argparse
import time
from numpy import nan 
import pandas as pd
from unidecode import unidecode
from Letterboxd import getLetterboxdMovieDetails
from Twitter import getTwitterData

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

    filmList = df['film'].to_list()
    yearList = df['year_film'].to_list()
    categoryList = df['category'].to_list()

    ratingList = []
    genreList = []
    sentiments = []
    subjectivity = []
    likes = []
    retweets = []
    
    start = time.time()
    
    for i, film in enumerate(filmList):
        if not pd.isnull(film):
            letterBoxdData = getLetterboxdMovieDetails(unidecode(film), yearList[i])
            twitterData = getTwitterData(unidecode(film))
            ratingList.append(letterBoxdData['rating'])
            genreList.append(letterBoxdData['genre'])
            sentiments.append(twitterData['sentiment'])
            subjectivity.append(twitterData['subjectivity'])
            likes.append(twitterData['likeCount'])
            retweets.append(twitterData['retweetCount'])
        else:
            ratingList.append('')
            genreList.append('')
            sentiments.append(nan)
            subjectivity.append(nan)
            likes.append(nan)
            retweets.append(nan)
        
        end = time.time()
        print("{} out of {}, time elapsed: {:.2f} seconds".format(i, len(filmList), end - start))

    end = time.time()
    print("Total time to rebuild data: {:.2f} seconds".format(end - start))

    df['rating'] = ratingList
    df['genre'] = genreList
    df['sentiment'] = sentiments
    df['subjectivity'] = subjectivity
    df['average likes'] = likes
    df['average retweets'] = retweets

    print(df)
    df.to_csv('oscar_nominees_full_columns.csv')

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