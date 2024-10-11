import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def train_and_save_model():
    # Load the dataset
    df = pd.read_csv('./data/news.csv')
    
    # Extract labels
    labels = df.label
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    # Initialize PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')
    
    # Save the model and vectorizer
    joblib.dump(pac, 'fake_news_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


def load_model():
    # Load the model and vectorizer from file
    if os.path.exists('fake_news_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    else:
        raise FileNotFoundError("Model or vectorizer file not found. Please train the model first.")


def classify_article(text):
    # Load the trained model and vectorizer
    model, vectorizer = load_model()
    
    # Transform the input text using the vectorizer
    text_transformed = vectorizer.transform([text])
    
    # Predict the label
    prediction = model.predict(text_transformed)
    return prediction[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake News Detection CLI')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Classify the given news article')
    
    args = parser.parse_args()
    
    if args.train:
        train_and_save_model()
    elif args.predict:
        result = classify_article(args.predict)
        print(f'Prediction: {result}')
    else:
        print("Please provide an argument. Use --train to train the model or --predict '<news_text>' to classify an article.")