from letterboxdpy import movie

def getLetterboxdMovieDetails(film, year):
    try:
        filmDetails = movie.Movie(film.lower())
        return filmDetails.rating.split(" ")[0]
    except:
        return ""

def getAllLetterboxdRatings(filmList, yearList):
    ratingList = []
    for i in range(len(filmList)):
        ratingList.append(getLetterboxdMovieDetails(filmList[i], yearList[i]))
    return ratingList

