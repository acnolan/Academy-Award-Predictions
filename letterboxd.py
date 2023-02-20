import pandas as pd
from numpy import nan
from letterboxdpy import movie

# Scrape letterboxd data for a course
def getLetterboxdMovieDetails(film, year):
    # Setup dictionary for the letterboxd data
    letterBoxdData = {}
    letterBoxdData['rating'] = nan
    letterBoxdData['genre'] = ''
    
    # Try to get the movie at the requested year
    # If it goes wrong just get the movie
    try:
        filmDetails = movie.Movie(film.lower(), year)
        letterBoxdData['rating'] = filmDetails.rating.split(" ")[0]
        letterBoxdData['genre'] = filmDetails.genres[0]
    except:
        try:
            filmDetails = movie.Movie(film.lower())
            letterBoxdData['rating'] = filmDetails.rating.split(" ")[0]
            letterBoxdData['genre'] = filmDetails.genres[0]
        except:
            print("Something went wrong with", film, " ", year, "!")

    if letterBoxdData['rating'] == 'None':
        letterBoxdData['rating'] = nan

    return letterBoxdData