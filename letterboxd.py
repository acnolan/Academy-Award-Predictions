import pandas as pd
from letterboxdpy import movie

# May need to add some logic to make sure it is the right year for the title
def getLetterboxdMovieDetails(film, year):
    # Setup dictionary for the letterboxd data
    letterBoxdData = {}
    letterBoxdData['rating'] = 0
    letterBoxdData['genre'] = 0

    try:
        filmDetails = movie.Movie(film.lower())
        letterBoxdData['rating'] = filmDetails.rating.split(" ")[0]
        letterBoxdData['genre'] = filmDetails.genres[0]
    except:
        print("Something went wrong with", film, " ", year, "!")

    return letterBoxdData
    

# Scraping Letterboxd data for thousands of movies is slow
# Possible optimization: cache movies since they get nominated more than once
def getAllLetterboxdRatings(filmList, yearList):
    ratingList = []
    genreList = []

    for i in range(len(filmList)):
        letterBoxdData = getLetterboxdMovieDetails(filmList[i], yearList[i])
        ratingList.append(letterBoxdData['rating'])
        genreList.append(letterBoxdData['genre'])
    
    return pd.DataFrame(list(zip(ratingList, genreList)), columns=['rating', 'genre'])

