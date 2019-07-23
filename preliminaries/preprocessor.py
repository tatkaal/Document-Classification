import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocessor(filename):
    df = pd.read_csv(filename, usecols=['paragraph', 'tag'])

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
    # words = stopwords.words("english")
    concated['cleaned'] = concated['paragraph'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())    
    return concated

# test = preprocessor('../Datasets/sentence.csv')
# print(test.shape)
# print(test.head(596))

