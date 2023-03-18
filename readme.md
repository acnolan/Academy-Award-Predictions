# Academy Award Prediction Tool
A basic data collection and prediction pipeline for the Academy Awards.

- Webscrape Letterboxd and Twitter to gather data
- Run a few basic Machine Learning algorithms to make predictions

A priori assumption: Past conditions still hold true for the Academy

## Predictions for 95th Academy Awards

| Category                      | My Prediction | Naive Bayes                       | KNN                                      | Logistic Regression               | Random Forest                     | SVM                                           |
|:------------------------------|:------------------------------|:----------------------------------|:-----------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------|
| actor in a leading role       | Austin Butler - Elvis | The Banshees of Inisherin         | Elvis                                    | The Banshees of Inisherin         | The Banshees of Inisherin         | Elvis                                         |
| actor in a supporting role    | Brendan Gleeson - The Banshees of Inisherin | Everything Everywhere All at Once | Everything Everywhere All at Once        | The Banshees of Inisherin         | Everything Everywhere All at Once | Causeway                                      |
| actress in a leading role     | Cate Blanchett - Tar | Everything Everywhere All at Once | Everything Everywhere All at Once        | To Leslie                         | Everything Everywhere All at Once | Tar                                           |
| actress in a supporting role  | Kerry Condon - The Banshees of Inisherin | Everything Everywhere All at Once | Everything Everywhere All at Once        | The Banshees of Inisherin         | Everything Everywhere All at Once | Black Panther: Wakanda Forever                |
| animated feature film         | Guillermo Del Toro's Pinocchio | Guillermo del Toro's Pinocchio    | Guillermo del Toro's Pinocchio           | Puss in Boots: The Last Wish      | Puss in Boots: The Last Wish      | Guillermo del Toro's Pinocchio                |
| cinematography                | All Quiet on the Western Front | All Quiet on the Western Front    | All Quiet on the Western Front           | Tar                               | All Quiet on the Western Front    | Bardo, False Chronicle of a Handful of Truths |
| costume design                | Everything Everywhere all at Once | Everything Everywhere All at Once | Everything Everywhere All at Once        | Everything Everywhere All at Once | Everything Everywhere All at Once | Black Panther: Wakanda Forever                |
| directing                     |  Everything Everywhere all at once | Everything Everywhere All at Once | Everything Everywhere All at Once        | Triangle of Sadness               | Everything Everywhere All at Once | The Banshees of Inisherin                     |
| documentary feature film      | Navalny | Fire of Love                      | All That Breathes                        | Fire of Love                      | All the Beauty and the Bloodshed  | All That Breathes                             |
| documentary short film        | The Elephant Whisperers | The Elephant Whisperers           | The Elephant Whisperers                  | The Elephant Whisperers           | Haulout                           | The Elephant Whisperers                       |
| film editing                  | Top Gun: Maverick | Everything Everywhere All at Once | Everything Everywhere All at Once        | Top Gun: Maverick                 | Everything Everywhere All at Once | Elvis                                         |
| international feature film    | All Quiet on the Western Front | All Quiet on the Western Front    | All Quiet on the Western Front           | The Quiet Girl                    | All Quiet on the Western Front    | Argentina, 1985                               |
| makeup and hairstyling        | Elvis | Black Panther: Wakanda Forever    | All Quiet on the Western Front           | The Batman                        | All Quiet on the Western Front    | Black Panther: Wakanda Forever                |
| music (original score)        | Babylon | Everything Everywhere All at Once | All Quiet on the Western Front           | Everything Everywhere All at Once | Everything Everywhere All at Once | Babylon                                       |
| music (original song)         | RRR | Everything Everywhere All at Once | Everything Everywhere All at Once        | RRR                               | Everything Everywhere All at Once | Black Panther: Wakanda Forever                |
| best picture                  | Everything Everywhere all at once | Everything Everywhere All at Once | All Quiet on the Western Front           | Top Gun: Maverick                 | The Banshees of Inisherin         | Avatar: The Way of Water                      |
| production design             | Babylon | Avatar: The Way of Water          | All Quiet on the Western Front           | Babylon                           | All Quiet on the Western Front    | Avatar: The Way of Water                      |
| short film (animated)         | Ice Merchants | My Year of Dicks                  | The Boy, the Mole, the Fox and the Horse | Ice Merchants                     | Ice Merchants                     | The Boy, the Mole, the Fox and the Horse      |
| short film (live action)      | Le Pupille | Le Pupille                        | An Irish Goodbye                         | The Red Suitcase                  | Le Pupille                        | An Irish Goodbye                              |
| sound                         | All Quiet on the Western Front | Avatar: The Way of Water          | All Quiet on the Western Front           | Top Gun: Maverick                 | All Quiet on the Western Front    | Avatar: The Way of Water                      |
| visual effects                | Avatar: Way of Water | Black Panther: Wakanda Forever    | All Quiet on the Western Front           | Top Gun: Maverick                 | All Quiet on the Western Front    | Avatar: The Way of Water                      |
| writing (adapted screenplay)  | Women Talking | All Quiet on the Western Front    | All Quiet on the Western Front           | Top Gun: Maverick                 | All Quiet on the Western Front    | All Quiet on the Western Front                |
| writing (original screenplay) | Everything Everywhere All at Once | Everything Everywhere all at once | Everything Everywhere All at Once | Everything Everywhere All at Once        | Triangle of Sadness               | Everything Everywhere All at Once | Everything Everywhere All at Once             |
| Training Accuracy | 60.86% | 66.15% | 79.68% | 79.68% | 80.27% | 79.68% |
| True Accuracy | 11/23 (47%) | 11/23 (47%) | 14/23 (61%) | 4/23 (17%) | 8/23 (35%) | 6/23 (26%) |

*Average accuracy of my guesses last three years

### Prediction Analysis
I will put some more detail in an upcoming blog post. But it looks like K-Nearest Neighbors won!

From the hyperparameter tuning, the best hyperparameters were:
- n neighbors: 20
- p: 1

The expected value of predictions is 4.5/23 if you guess completely randomly (1/5 * 22 + 1/10). All categories have 5 nominees except best picture which has 10. Therefore, all of these models did better than true random except Logistic Regression.

Next year we may want to focus in on KNN as it performed the best.

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