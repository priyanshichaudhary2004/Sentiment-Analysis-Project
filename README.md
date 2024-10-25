Overview
This Python script implements a Flask web application for sentiment analysis of product reviews. It processes user-submitted reviews and predicts their sentiment using Natural Language Processing (NLP) and machine learning techniques.

Key Components
Flask Framework: Serves as the backbone of the web application, handling routes and rendering templates.

Data Preprocessing:

Loads a CSV dataset of product reviews.
Cleans and processes the review text by removing punctuation, numbers, and stopwords, and performs lemmatization.
Sentiment Labeling:

Classifies reviews into three categories based on ratings:
Positive (ratings > 3.0)
Negative (ratings < 3.0)
Neutral (ratings = 3.0)
Machine Learning Model:

Uses TF-IDF vectorization to convert text data into numerical format.
Implements a Logistic Regression classifier, trained on a balanced dataset using SMOTE to ensure fair representation of sentiments.
Web Routes:

/: Displays the main page where users can input their reviews.
/result: Processes the submitted review and displays the predicted sentiment.
Dependencies
This script requires the following Python libraries:

Flask
pandas
numpy
nltk
scikit-learn
imbalanced-learn
