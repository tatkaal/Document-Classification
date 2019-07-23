import os
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from preliminaries.preprocessor import preprocessor
from nltk.stem import WordNetLemmatizer

df = preprocessor('Datasets/textclassify.csv')
print(df.tag.value_counts())
# uniqueTags = pd.DataFrame(df.tag.unique()).values
# numCount = df.tag.value_counts()
# rc = df.shape
# sample = df.head(5)
# print(uniqueTags, numCount,rc,sample)

#creates new column cat_id in the dataframe
df['cat_id'] = df['tag'].factorize()[0]
cat_id_df = df[['tag', 'cat_id']].drop_duplicates()

#Dictionary
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'tag']].values)

#To visualize the dataset distribution
fig = plt.figure(figsize=(10,8))
colors = ['red', 'green', 'blue', 'yellow', 'black']
df.groupby('tag').cleaned.count().sort_values().plot.barh(ylim=0, color=colors, title="Number of sentence in each tag\n")
plt.xlabel('No. of occurences', fontsize = 8)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4, ngram_range=(1,2))
#transforms each row in vector
features = tfidf.fit_transform(df.cleaned).toarray()
labels = df.cat_id
print("Each of the %d sentences are represented by %d features (TF-IDF score of unigrams and bigrams)"%(features.shape))

##To check the most corelated terms for each categories
N = 8
for tag, cat_id in sorted(cat_to_id.items()):
  features_chi2 = chi2(features, labels == cat_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(tag))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

X = df['cleaned'] # Collection of documents
y = df['tag'] # Target or the labels we want to predict (i.e., the 5 different categories of a document)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

#models list for cross validation
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

##Comparing model performance
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print(acc)


##Model Evaluation
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Classification report
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_test, y_pred, 
                                    target_names= df['tag'].unique()))

##visualizing confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(2,2))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=cat_id_df.tag.values, 
            yticklabels=cat_id_df.tag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)
plt.show()

##To check which of the sentences are wrongly predicted
# for predicted in cat_id_df.cat_id:
#   for actual in cat_id_df.cat_id:
#     if predicted != actual and conf_mat[actual, predicted] >= 2:
#       print("'{}' predicted as '{}' : {} examples.".format(id_to_cat[actual], 
#                                                            id_to_cat[predicted], 
#                                                            conf_mat[actual, predicted]))
    
#       display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['tag', 
#                                                                 'cleaned']])
#       print('')

##To predict on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4,
                        ngram_range=(1, 2))

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)

sent = "As broad as their sounds are, there are several very distinct similarities and differences between electric and acoustic guitars. For instance, both utilize the use of a body for the neck to attach to and a neck with frets for finger placement. The strings attach to the lower end of the body and go all the way to the head, or the top of the neck. They both use strings that vary in gauge, or size, which are vital to produce sound when they are picked, hammered on, or strummed as a group. Similarly, each is tuned in the same manner to produce the proper tone desired. An acoustic guitar needs no amplifier to make its sound loud enough to be heard. An acoustic guitar uses the body of the guitar as its amplifier. Because the body is very thick and hollow it is able to project its own natural sound loudly. This makes it very portable and capable of being played virtually anywhere. An acoustic guitar doesn’t need any foot pedals, volume and tone knobs, or any other hardware like that to produce the sound it makes. An electric guitar is very hard to hear without an amplifier. An electric guitar requires the use of an amplifier to transport the sound though pickups that are secured in the body. These sounds are transferred through a cable connected to the guitar. The cable then goes to the amplifier which produces the sounds out of the speakers. Volume and tone knobs on the electric guitar can make it louder or change the sound of the strings being played. Additionally, foot pedals can be added to produce even more different sounds so that the musical capabilities of the electric guitar are almost limitless. There’s not a lot of music that I listen to that doesn’t have some sort of electric or acoustic guitar in the mix, either as the main instrument or as small as a fill in for a certain sound. The genre of the music frequently dictates which type of guitar should be used."

# lemmatizer = WordNetLemmatizer()

# df_test = pd.DataFrame(columns=['test'])
# df_test['test'].append(sent, columns = ['test'])
# norm = df_test['test'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())    
print(model.predict(fitted_vectorizer.transform([sent])))

##To check the real label
# print(df[df['cleaned'] == norm])