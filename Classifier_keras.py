import numpy as np
import pandas as pd
from scipy.stats import chi
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras.utils import to_categorical
from pathlib import Path

import DataEditer
from DataEditer import DataEdit



data = DataEditer.readData()
resumes = data["Resume"].values
labels = data["Category"].values

resumes = DataEdit.stringRemoverAndStopword(resumes)
resumes = DataEdit.stemAndStopText(resumes)

X_train, X_test, Y_train, Y_test = train_test_split(resumes, labels, test_size=0.2)
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True, min_df=1, use_idf=True)
x = vectorizer.fit_transform(X_train, Y_train)
y = vectorizer.fit_transform(X_test, Y_test)
predict_vector = vectorizer.fit_transform({'Steve Pomeroy\nResume\nEducation\nFall 2000\xe2\x80\x93Fall\n2005\n\nBachelor of Science, Rochester Institute of Technology, Rochester, NY.\nReceived a Bachelors of Science in Computer Science from the Rochester Institute of\nTechnology. Concentrated in computer security, computer language construction, Japanese\nand psychology.\n\nExperience\nApril 2013\xe2\x80\x93 Senior Software Engineer, LevelUp.\npresent Developed Android application software, focusing on SDK design and development.\n2007\xe2\x80\x932013\n\nSystems Designer & Lead Mobile Developer, MIT Mobile Experience Lab.\nLead the development of and participated in the design of numerous projects. Functioned\nas lead developer for most of their implementations. Duties in projects mostly centered\naround Android app development, but also'})

unique_labels = set(labels)

train_data = keras.preprocessing.sequence.pad_sequences(x.toarray(),

                                                        padding='post',
                                                        maxlen=4096)

test_data = keras.preprocessing.sequence.pad_sequences(y.toarray(),

                                                       padding='post',
                                                       maxlen=4096)

predict_data = keras.preprocessing.sequence.pad_sequences(predict_vector.toarray(), padding='post', maxlen=4096)

le = LabelEncoder()
labels_train = to_categorical(le.fit_transform(Y_train))
print(labels_train)
labels_test = to_categorical(le.fit_transform(Y_test))

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(1350, 25))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(40, activation=tf.nn.relu))
model.add(keras.layers.Dense(20, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(25, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
              loss='binary_crossentropy',
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

print(results)
unique_labels = sorted(unique_labels)
i = 0
for x in range(0,3):
    predicts = model.predict_classes(predict_data)
    print(unique_labels)
    print(unique_labels[predicts[0]])



# model.save_weights('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\my_checkpoint', overwrite=True)
