from textblob import TextBlob

# Gets the sentiment polarity score (-1 to 1) of text
def conductSentimentAnalysis(text):
    blob = TextBlob(text)
    return blob.sentiment