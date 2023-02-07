import pandas as pd
from letterboxd import getAllLetterboxdRatings
#import sklearn

df = pd.read_csv("./oscar_nominees.csv")

filmList = df['film'].to_list()
yearList = df['year_film'].to_list()

df['letterboxdRating'] = getAllLetterboxdRatings(filmList,yearList)

print(df.head(5))