# Academy Award Prediction Tool
A basic data collection and prediction pipeline for the Academy Awards.

- Webscrape Letterboxd and Twitter to gather data
- Run a few basic Machine Learning algorithms to make predictions

A priori assumption: Past conditions still hold true for the Academy

## Running the code

### Only run machine learning
- Make sure test.csv and train.csv are in the same directory as AcademyAwardPrediction.py
- Run `python AcademyAwardPrediction.py`

### Run machine learning with data visualizations
- Use the -v flag `python AcademyAwardPrediction.py -v`

### Rebuild the dataset
- Run `python AcademyAwardPrediction.py -b`
- The -b flag runs code to webscrape and rebuild the dataset
- Note: this takes a long time, about 6 hours

## Future Enhancements
This code is very basic and can be optimized in the future. Once we see how it does this year we can make changes to do even better in the next.

- Rotten Tomatos: Audience vs Critic scores
- Collect data from web search like google or bing
- Collect data from other social media (Reddit?)
- Train models for different categories instead of general
- Try out a neural net or some deep learning
- Fix overfitting on some models

## References

### Packages Used
- [letterboxdpy](https://pypi.org/project/letterboxdpy/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [snscrape](https://github.com/JustAnotherArchivist/snscrape)
- [textblob](https://pypi.org/project/textblob/)
- [unidecode](https://pypi.org/project/Unidecode/)

### Blogs, Videos, and Guides
1. [Oscar movie Dataset](https://www.kaggle.com/datasets/unanimad/the-oscar-award)
2. [A Quick Guide To Sentiment Analysis | Sentiment Analysis In Python Using Textblob | Edureka](https://www.youtube.com/watch?v=O_B7XLfx0ic)
3. [Scrape Twitter with 5 Lines of Code](https://www.youtube.com/watch?v=PUMMCLrVn8A)
4. [Introduction to Data Preprocessing](https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d)
5. [Naive Bayes Classifier - Multinomial Bernoulli Gaussian Using Sklearn in Python - Tutorial 32](https://www.youtube.com/watch?v=ok2s1vV9XW0&t=614s&ab_channel=codebasics)
6. [Machine Learning Tutorial Python - 11 Random Forest](https://www.youtube.com/watch?v=99MN-rl8jGY&t=601s&ab_channel=TEW22)