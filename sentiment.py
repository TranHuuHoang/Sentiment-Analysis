#!/usr/bin/env python
import keras
import numpy as np
import pandas as pd
import os
import glob
import math
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D,SimpleRNN,LeakyReLU
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords


np.random.seed(8)

def loadTrainData(path):     #loads data for training.
    D = pd.read_csv(path)
    feature_names  = np.array(list(D.columns.values))
    X_cat = np.array(list(D['Summary'][:50000]))
    X_train = np.array(list(D['Text'][:50000]))
    for x in range(0,X_cat.shape[0]):
         X_train[x] = X_cat[x] + " " + X_train[x]      
    Y_train = np.array(list(D['Score'][:50000]))
    # Make score from 0 to 4
    for y in range(0,Y_train.shape[0]):
        Y_train[y] = Y_train[y] - 1
    return  X_train, Y_train, feature_names

def loadTestData(path):     #loads data for testing.
    D = pd.read_csv(path)
    X_test=np.array(list(D['Text'][55000:60000]))
    X_cat = np.array(list(D['Summary'][55000:60000]))
    for x in range(0,X_cat.shape[0]):
         X_test[x] = X_cat[x] + " " + X_test[x]
    Y_test = np.array(list(D['Score'][55000:60000]))
    # Make score from 0 to 4
    for y in range(0,Y_test.shape[0]):
        Y_test[y] = Y_test[y] - 1
    return  X_test, Y_test


def shuffle(a, b): # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]


X_train, Y_train, feature_names = loadTrainData('./Reviews.csv')
X_test, Y_test = loadTestData('./Reviews.csv')
print('Training data shapes')
print('X_train.shape: ', X_train.shape)
print('Y_train.shape: ',Y_train.shape)


Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))
# Tokenizer.fit_on_texts(X_train)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
print("Vocab size",Tokenizer_vocab_size)

#masking
num_test = 5000
mask = range(num_test)

Y_Val = Y_train[:num_test]
X_Val = X_train[:num_test]


X_train = X_train[num_test:]
Y_train = Y_train[num_test:]

'''
#Removing stopwords result in lower accuracy?
stop_words = stopwords.words('english')

def removeStopWords(data, stop_words):
    count = 0
    print("Process:\n")
    for i in range (len(data)):
        data[i] = " ".join([w for w in data[i].split() if w not in stop_words])
        count = count + 1
        print(count, end=" ")
    return data

print("\nBefore:\n")
print(X_train[0])
print(X_Val[0])
print(X_test[0])

print("\nRemoving stopwords...")
X_train = removeStopWords(X_train, stop_words)
X_val = removeStopWords(X_Val, stop_words)
X_test = removeStopWords(X_test, stop_words)
print("Successfully remove stop words!")

print("\nAfter:\n")
print(X_train[0])
print(X_Val[0])
print(X_test[0])
'''

#Encode text
maxWordCount= 1000
maxDictionary_size = Tokenizer_vocab_size

encoded_Xtrain = Tokenizer.texts_to_sequences(X_train)
encoded_Xval = Tokenizer.texts_to_sequences(X_Val)
encoded_Xtest = Tokenizer.texts_to_sequences(X_test)


#padding all text to same size
X_Train_encodedPadded = sequence.pad_sequences(encoded_Xtrain, maxlen=maxWordCount)
X_Val_encodedPadded = sequence.pad_sequences(encoded_Xval, maxlen=maxWordCount)
X_test_encodedPadded = sequence.pad_sequences(encoded_Xtest, maxlen=maxWordCount)

Y_train = keras.utils.to_categorical(Y_train, 5)
Y_Val   = keras.utils.to_categorical(Y_Val, 5)
	

shuffle(X_Train_encodedPadded,Y_train)

print('Feature ',feature_names)
print(' After extracting a validation set of '+ str(num_test)+'  ')
print(' Training data shapes ')
print('X_train.shape is ', X_train.shape)
print('Y_train.shape is ',Y_train.shape)
print(' Validation data shapes ')
print('Y_Val.shape is ',Y_Val.shape)
print('X_Val.shape is ', X_Val.shape)
print(' Test data shape ')
print('X_test.shape is ', X_test.shape)

print(' After padding all text to same size of '+ str(maxWordCount)+' ')
print(' Training data shapes ')
print('X_train.shape is ', X_train.shape)
print('Y_train.shape is ',Y_train.shape)
print(' Validation data shapes ')
print('Y_Val.shape is ',Y_Val.shape)
print('X_Val.shape is ', X_Val.shape)
print(' Test data shape ')
print('X_test.shape is ', X_test.shape)

#model
model = Sequential()

model.add(Embedding(maxDictionary_size, 100, input_length=maxWordCount)) #to change words to ints
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#hidden layers
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
#model.add(SimpleRNN(30))
# model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu',W_constraint=maxnorm(1)))
model.add(LeakyReLU(alpha=0.01))
# model.add(Dropout(0.6))
#model.add(Dense(50, activation='relu',W_constraint=maxnorm(1)))

# model.add(Dropout(0.5))
 #output layer
model.add(Dense(5, activation='softmax'))

# Compile model
# adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

model.summary()


learning_rate=0.01
epochs = 4
batch_size = 50
#sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_25', histogram_freq=0, write_graph=True, write_images=False)

checkpointer = ModelCheckpoint(filepath="./weights/weights_25.hdf5", verbose=1, save_best_only=True, monitor="val_loss")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

print(("Training"))

history  = model.fit(X_Train_encodedPadded, Y_train, epochs = epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val_encodedPadded, Y_Val), callbacks=[tensorboard, reduce_lr, checkpointer, earlyStopping])

print(("Test Score"))

scores = model.evaluate(X_Val_encodedPadded, Y_Val, verbose=0)
print("Validation Accuracy: %.2f%%" % (scores[1]*100))
print(("Predicting"))

predicted_classes = model.predict_classes(X_test_encodedPadded, batch_size=batch_size, verbose=1)

'''
f = open('result.csv', 'w')
f.write('itemID,Category\n')


for i in range(0,X_test_itemID.shape[0]):
    f.write(str(X_test_itemID[i])+","+str(predicted_classes[i])+'\n')

f.close()
'''
test_score = accuracy_score(Y_test, predicted_classes)
print("Test accuracy: %.2f%%" % (test_score*100))