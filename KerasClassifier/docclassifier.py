import numpy as np # for linear algebra
import pandas as pd # data processing
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from nltk.tokenize import sent_tokenize

data = pd.read_csv('Datasets/textclassify.csv', usecols=['paragraph', 'tag'])
print(data.tag.value_counts())

shuffled = data.reindex(np.random.permutation(data.index))
descriptive = shuffled[shuffled['tag'] == 'descriptive']
comparative = shuffled[shuffled['tag'] == 'comparative']
causeEffect = shuffled[shuffled['tag'] == 'cause and effect']
problemSolution = shuffled[shuffled['tag'] == 'problem and solution']
sequential = shuffled[shuffled['tag'] == 'sequential']

concated = pd.concat([descriptive, comparative, causeEffect, problemSolution, sequential], ignore_index=True)
#Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))
concated['LABEL'] = 0

concated.loc[concated['tag'] == 'descriptive', 'LABEL'] = 0
concated.loc[concated['tag'] == 'comparative', 'LABEL'] = 1
concated.loc[concated['tag'] == 'cause and effect', 'LABEL'] = 2
concated.loc[concated['tag'] == 'problem and solution', 'LABEL'] = 3
concated.loc[concated['tag'] == 'sequential', 'LABEL'] = 4

# print(concated['LABEL'])

labels = to_categorical(concated['LABEL'], num_classes=5)
# print(labels)
if 'tag' in concated.keys():
    concated.drop(['tag'], axis=1)
'''
 [1. 0. 0. 0. 0.] descriptive
 [0. 1. 0. 0. 0.] comparative
 [0. 0. 1. 0. 0.] cause and effect
 [0. 0. 0. 1. 0.] problem and solution
 [0. 0. 0. 0. 1.] sequential
'''

n_most_common_words = 65
max_len = 116
tokenizer = Tokenizer(n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(concated['paragraph'].values)
sequences = tokenizer.texts_to_sequences(concated['paragraph'].values)
# print(min(sequences))
# print(len(min(sequences)))
# print(max(sequences))
# exit()
word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)
# print(labels[:2])
# exit()

X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.2, random_state=42)

epochs = 50
emb_dim = 128
batch_size = 2
# labels[:2]

# print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(20, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2) #,callbacks=[EarlyStopping(monitor='val_loss',patience=10, min_delta=0.0001)]

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

model.save('model.h5')

# import matplotlib.pyplot as plt

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

txt = ["Standing on his hind legs, this rare andalucian stallion is fearless. His ears are turned back while his noble looking head is held high. His all black coat glistens in the late afternoon sun. His face displays a strong confidence with his nostrils flared, his veins bulging from his cheek bones, and his fiery black eyes burning holes into the souls of those who stare into them. His neck muscles are tensed and thickened with adrenalin. His black main is thrown into the wind like a flag rippling in the winds of a tornado. His muscular front legs are brought up to his chest displaying his flashing gray hooves that could crush a man's scull with one blow. His backbone and underbelly are held almost straight up and his hind quarters are tensed. His back legs are spread apart for balance. His back hooves are pressed into the earth; therefore, his hooves cause deep gouges from the weight of his body on the soil. His black tail is held straight down and every once in a while a burst of wind catches it and then it floats down back into place like an elegant piece of silk falling from the sky. His bravery and strength are what made his breed prized as a warhorse."]

#For testing paragraphs
seq = tokenizer.texts_to_sequences([txt])
# print(seq)
padded = pad_sequences(seq, maxlen=116)
print(padded)
# print(max(padded))
# print(len(max(padded)))
# print(min(padded))
# print(len(min(padded)))
pred = model.predict(padded)
# print(np.argmax(pred))
labels = ['descriptive', 'comparative', 'cause and effect', 'problem and solution', 'sequential']
# temp = 
print(pred, labels[np.argmax(pred)])

# #Testing for sentences
# convtostr = ''.join(txt)
# filTxt = sent_tokenize(convtostr)
# # print(filTxt)
# for i in filTxt:
#     print(i)
#     pred=0
#     seq = tokenizer.texts_to_sequences([i])
#     # print(seq)
#     padded = pad_sequences(seq, maxlen=88)
#     pred = model.predict(padded)
#     # print(np.argmax(pred))
#     labels = ['descriptive', 'comparative', 'cause and effect', 'problem and solution', 'sequential']
#     print(pred, labels[np.argmax(pred)])








