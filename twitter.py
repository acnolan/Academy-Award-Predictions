import pandas as pd
import snscrape.modules.twitter as sntwitter
from SentimentAnalysis import conductSentimentAnalysis

# How many tweets to get
numberOfTweets = 10

# Get twitter data for a course
def getTwitterData(film):
    # Setup counters for the film
    twitterData = {}
    twitterData['sentiment'] = 0
    twitterData['subjectivity'] = 0
    twitterData['likeCount'] = 0
    twitterData['retweetCount'] = 0

    # Analyze 50 tweet results about the film name + oscars
    scraper = sntwitter.TwitterSearchScraper(film + " oscars")
    for i, tweet in enumerate(scraper.get_items()):
        twitterData = incrementTwitterDataValues(tweet, twitterData)
        if i == numberOfTweets/2:
            break

    # Also do 50 tweets of film name + Academy Awards
    scraper = sntwitter.TwitterSearchScraper(film + " academy awards")
    for i, tweet in enumerate(scraper.get_items()):
        twitterData = incrementTwitterDataValues(tweet, twitterData)
        if i == numberOfTweets/2:
            break
    
    return twitterData

# Reusable function to increment the dictionary values
def incrementTwitterDataValues(tweet, twitterData):
    sentimentData = conductSentimentAnalysis(tweet.rawContent)
    twitterData['sentiment'] += sentimentData.polarity / numberOfTweets
    twitterData['subjectivity'] += sentimentData.subjectivity / numberOfTweets
    twitterData['likeCount'] += tweet.likeCount / numberOfTweets
    twitterData['retweetCount'] += tweet.retweetCount / numberOfTweets
    return twitterData