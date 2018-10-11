import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras.utils import to_categorical

from DataEditer import DataEdit


data = pd.read_csv('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\resume_dataset.csv')
resumes = data["Resume"].values
labels = data["Category"].values

resumes = DataEdit.stringRemoverAndStopword(resumes)
resumes = DataEdit.stemAndStopText(resumes)

X_train, X_test, Y_train, Y_test = train_test_split(resumes, labels, test_size=0.2)
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)
x = vectorizer.fit_transform(X_train)
y = vectorizer.fit_transform(X_test)

train_data = keras.preprocessing.sequence.pad_sequences(x.toarray(),

                                                        padding='post',
                                                        maxlen=1024)

test_data = keras.preprocessing.sequence.pad_sequences(y.toarray(),

                                                       padding='post',
                                                       maxlen=1024)

le = LabelEncoder()
labels_train = to_categorical(le.fit_transform(Y_train))
labels_test = to_categorical(le.fit_transform(Y_test))

print(labels_train, labels_test)
print(train_data[0])

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(25, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:100]
partial_x_train = train_data[100:]

y_val = labels_train[:100]
partial_y_train = labels_train[100:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=50,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, labels_test)