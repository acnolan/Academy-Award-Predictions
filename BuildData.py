import string
import time
from numpy import nan 
import pandas as pd

from unidecode import unidecode
from Letterboxd import getLetterboxdMovieDetails
from Twitter import getTwitterData

# Count the frequencies of the film and nominees occuring
def countNominationFrequency(filmList, nameList):
    # Set up the frequency dicts
    films = {}
    nominees = {}

    # Count
    for i, film in enumerate(filmList):
        if film in films:
            films[film] += 1
        else:
            films[film] = 1
            
        if nameList[i] in nominees: 
            nominees[nameList[i]] += 1
        else: 
            nominees[nameList[i]] = 1
    
    return films, nominees

# Build the data table
def rebuildTable():
    df = pd.read_csv("./oscar_nominees.csv")

    filmList = df['film'].to_list()
    yearList = df['year_film'].to_list()
    nameList = df['name'].to_list()

    ratingList = []
    genreList = []
    sentiments = []
    subjectivity = []
    likes = []
    retweets = []
    filmCount = []
    nomineeCount = []

    # Store movies we have already seen (so we don't have to webscrape for them multiple times)
    filmDict = {}

    # Store counts of nominees
    filmFreq, nomineeFreq = countNominationFrequency(filmList, nameList)

    start = time.time()
    
    for i, film in enumerate(filmList):
        if not pd.isnull(film):
            if film in filmDict:
                letterBoxdData = filmDict[film]['letterBoxdData']
                twitterData = filmDict[film]['twitterData']
                filmDict[film]['count'] += 1
            else:
                letterBoxdData = getLetterboxdMovieDetails(unidecode(film).translate(str.maketrans('', '', string.punctuation)), yearList[i])
                twitterData = getTwitterData(unidecode(film))
                filmDict[film] = {}
                filmDict[film]['letterBoxdData'] = letterBoxdData
                filmDict[film]['twitterData'] = twitterData 
                filmDict[film]['count'] = 1

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
        
        filmCount.append(filmFreq[film])
        nomineeCount.append(nomineeFreq[nameList[i]])
        
        end = time.time()
        print("{} out of {}, time elapsed: {:.2f} seconds".format(i+1, len(filmList), end - start))

    end = time.time()
    print("Total time to rebuild data: {:.2f} seconds".format(end - start))

    # Add columns to the data table
    df['rating'] = ratingList
    df['genre'] = genreList
    df['sentiment'] = sentiments
    df['subjectivity'] = subjectivity
    df['average_likes'] = likes
    df['average_retweets'] = retweets
    df['film_count'] = filmCount
    df['nominee_count'] = nomineeCount

    df.loc[df['ceremony'] == 95].to_csv('test.csv')
    df.loc[df['ceremony'] != 95].to_csv('train.csv')