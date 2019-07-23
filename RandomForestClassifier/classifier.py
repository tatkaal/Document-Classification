import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import pickle
import sys
 
from preliminaries.preprocessor import preprocessor

df = preprocessor('Datasets/textclassify.csv')
vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(df['cleaned']).toarray()
print(final_features.shape)
print(len(final_features))
# exit()
# print(final_features.shape) #65 features from 135

#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = df['cleaned']
Y = df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=750)),
                     ('clf', RandomForestClassifier())])

# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)

ytest = np.array(y_test)

# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))

# vectorizer = model.named_steps['vect']
# chi = model.named_steps['chi']
# clf = model.named_steps['clf']

# feature_names = vectorizer.get_feature_names()
# feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
# feature_names = np.asarray(feature_names)
# print(feature_names)

# in this case, I have 5 different classes:
# target_names = ['descriptive', 'comparative', 'cause and effect', 'problem and solution', 'sequential']
# print("top 10 keywords per class:")
# for i, label in enumerate(target_names):
#     top10 = np.argsort(clf.feature_importances_)[-10:]
#     print("%s: %s" % (label, " ".join(feature_names[top10])))

print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(["There are several characteristics which distinguish plants from animals. Green plants are able to manufacture their own food from substances in the environment. This process is known as photosynthesis. In contrast, animals, including man, get their food either directly from plants or indirectly by eating animals which have eaten plants. Plants are generally stationary. Animals, on the other hand, can usually move about. In external appearance, plants are usually green. They grow in a branching fashion at their extremities, and their growth continues throughout their lives. Animals, however, are very diverse in their external appearance. Their growth pattern is not limited to their extremities. It is evenly distributed and only occurs in a definite time period. Therefore, the differences between plants and animals is quite significant."]))