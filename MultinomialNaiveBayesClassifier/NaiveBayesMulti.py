import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

df = pd.read_csv('Datasets/sentence.csv')
# df = df[pd.notnull(df['tag'])]
# print(df.head(10))
# print(df['paragraph'].apply(lambda x: len(x.split(' '))).sum())
class_tags = ['descriptive', 'comparative','cause and effect', 'problem and solution', 'sequential']
print(df.tag.value_counts())

# def print_plot(index):
#     example = df[df.index == index][['paragraph', 'tag']].values[0]
#     if len(example) > 0:
#         print(example[0])
#         print('Tag:', example[1])

# print_plot(0)

shuffled = df.reindex(np.random.permutation(df.index))
descriptive = shuffled[shuffled['tag'] == 'descriptive']
comparative = shuffled[shuffled['tag'] == 'comparative']
causeEffect = shuffled[shuffled['tag'] == 'cause and effect']
problemSolution = shuffled[shuffled['tag'] == 'problem and solution']
sequential = shuffled[shuffled['tag'] == 'sequential']

concated = pd.concat([descriptive, comparative, causeEffect, problemSolution, sequential], ignore_index=True)
#Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))

lemmatizer = WordNetLemmatizer()
words = stopwords.words("english")
concated['cleaned'] = concated['paragraph'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = concated.cleaned
Y = concated.tag
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

##Naive Bayes Classifier for multinomial models
model = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, labels=class_tags, target_names=class_tags))

print(model.predict(["I bought a new house in Switzerland."]))

# print(set(y_test)-set(y_pred)) #checks whether y-test is in y-pred or not