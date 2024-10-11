import os
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def detect(file_path):
    # read the news data in to a data frame
    df = pd.read_csv('./data/news.csv')

    # get shape and head
    shape = df.shape
    head = df.head()

    # labels from the data
    labels = df.label
    head_labels = labels.head()

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

    # initalize TfidfVectorizor
    tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)

    # fit and transform train set, and transform test set
    tfidf_train = tfidf_vect.fit_transform(x_train)
    tfidf_test = tfidf_vect.transform(x_test)

    # initalize PassiveAgressiveClassifer
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')

    # build confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print(f'{cm}\n')
    
def main():
    # loop through all data sets and run the fake news detect algorithm on each set
    for file in os.listdir('./data'):
        filename = os.fsdecode(file)
        if file.endswith('.csv'):
            print(f'Detecting Fake News in {filename}...')
            detect(f'./data/{filename}')
    
    
if __name__ == "__main__":
    main()