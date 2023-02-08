import pandas as pd
from letterboxd import getAllLetterboxdRatings
from twitter import getAllTwitterData
#import sklearn

df = pd.read_csv("./oscar_nominees.csv")

df2 = df.head(5)

filmList = df2['film'].to_list()
yearList = df2['year_film'].to_list()
categoryList = df2['category'].to_list()

df2 = pd.concat([df2, getAllLetterboxdRatings(filmList,yearList)], axis=1)
df2 = pd.concat([df2, getAllTwitterData(filmList)], axis=1)

print(df2.head(5))