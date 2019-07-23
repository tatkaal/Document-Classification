import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

from preliminaries.preprocessor import preprocessor

concated = preprocessor('Datasets/sentence.csv')
# print(concated.tag.value_counts())
# print(pd.DataFrame(concated.unique()).values) ###To get distinct categories
# concated = concated[pd.notnull(concated['tag'])] #to remove missing values
# print(concated.shape)
# print(concated.head(4))
# exit()

train_size = int(len(concated) * .8)
train_posts = concated['cleaned'][:train_size]
train_tags = concated['tag'][:train_size]

test_posts = concated['cleaned'][train_size:]
test_tags = concated['tag'][train_size:]

max_words = 200
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 1
epochs = 50

# Build the model
model = Sequential()
model.add(Dense(1, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.03))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
txt = ["I am walking on the road."]
pre = tokenize.texts_to_matrix([txt])
val = model.predict(pre)
labels = ['descriptive', 'comparative', 'cause and effect', 'problem and solution', 'sequential']
print(val, labels[np.argmax(val)])
print()