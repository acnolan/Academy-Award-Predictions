import pandas as pd
from letterboxdpy import movie

# Scrape letterboxd data for a course
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