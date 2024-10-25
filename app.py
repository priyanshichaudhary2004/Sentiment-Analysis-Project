from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset and preprocessing
dataset = pd.read_csv("Instruments_Reviews.csv")
dataset['reviewText'] = dataset['reviewText'].fillna(value="")
dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]

# Text cleaning function
def Text_Cleaning(Text):
    Text = Text.lower()
    punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    Text = Text.translate(punc)
    Text = re.sub(r'\d+', '', Text)
    Text = re.sub(r'https?://\S+|www\.\S+', '', Text)  # Fixed: Use same variable name
    Text = re.sub('\n', '', Text)
    return Text

# Text processing function
Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])

def Text_Processing(Text):
    Processed_Text = []
    Lemmatizer = nltk.stem.WordNetLemmatizer()
    Tokens = nltk.word_tokenize(Text)
    for word in Tokens:
        if word not in Stopwords:
            Processed_Text.append(Lemmatizer.lemmatize(word))
    return " ".join(Processed_Text)

# Apply cleaning and processing to the reviews
dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Cleaning(Text))
dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Processing(Text))

# Labeling based on ratings
def Labelling(Rows):
    if Rows["overall"] > 3.0:
        return "Positive"
    elif Rows["overall"] < 3.0:
        return "Negative"
    else:
        return "Neutral"

dataset["sentiment"] = dataset.apply(Labelling, axis=1)

# Drop unnecessary columns
Columns = ["reviewerID", "asin", "reviewerName", "helpful", "unixReviewTime", "reviewTime"]
dataset.drop(columns=Columns, axis=1, inplace=True)

# Encode the sentiment
Encoder = LabelEncoder()
dataset["sentiment"] = Encoder.fit_transform(dataset["sentiment"])

# TF-IDF Vectorization (adjusted to unigrams and bigrams)
TF_IDF = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = TF_IDF.fit_transform(dataset["reviews"])
y = dataset["sentiment"]

# Balance data using SMOTE
Balancer = SMOTE(random_state=42)
X_final, y_final = Balancer.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=42)

# Train Logistic Regression model
Classifier = LogisticRegression(random_state=42, C=6866.488450042998, penalty='l2')
Classifier.fit(X_train, y_train)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        review = request.form['review']
        print(review)
        sentiment = predict_sentiment(review)  # Call the prediction function
        print(sentiment)
        sentiment_label = Encoder.inverse_transform([sentiment])[0]  # Decode the label
        print(sentiment_label)
        return render_template('result.html', sentiment=sentiment_label, user_review=review)

def predict_sentiment(review):
    if len(review.strip()) == 0:  # Handle empty reviews
        return Encoder.transform(["Neutral"])[0]  # Default to neutral sentiment for empty reviews

    cleaned_review = Text_Cleaning(review)
    processed_review = Text_Processing(cleaned_review)

    # Check if the processed review is empty after cleaning
    if len(processed_review.strip()) == 0:
        return Encoder.transform(["Neutral"])[0]  # Default to neutral sentiment if processing removes all text

    review_vectorized = TF_IDF.transform([processed_review])
    sentiment_prediction = Classifier.predict(review_vectorized)
    return sentiment_prediction[0]  # Return the encoded sentiment

if __name__ == '__main__':
    app.run(debug=True,port=5001)
