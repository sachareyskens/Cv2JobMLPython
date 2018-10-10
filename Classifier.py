import string

import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2



def stringRemoverAndStopword(l):
    r = []

    filter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2'
        , '3', '4', '5', '6', '7', '8', '9']
    table = str.maketrans({key: None for key in string.punctuation})
    q=""
    for s in l:
        s = s.replace("\\n", "")
        s = s.replace("b'", "")
        s = s.replace("'", "")
        s = s.replace('b"', '')
        s = s.replace('"', '')
        s = s.replace("page", "")
        for x in filter_list:
            for y in filter_list:
                s = s.replace("\\x" + x + y, "")


        s = s.translate(table)
        r.append(s.lower())
    return r


def stemAndStopText(r):
    stemmer = SnowballStemmer('english')
    words = stopwords.words('english')
    l = []
    for s in r:
        q = ""
        for w in s.split(" "):
            if w not in words:
                q+=w + " "
        l.append(q)
        for s in l:
            s = s.replace("  ", " ")
    return l

data = pd.read_csv('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\resume_dataset.csv')

resumes = data["Resume"].values
labels = data["Category"].values


resumes = stringRemoverAndStopword(resumes)
resumes = stemAndStopText(resumes)


X_train, X_test, Y_train, Y_test = train_test_split(resumes, labels, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

model = pipeline.fit(X_train, Y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = set(labels)

for i, label in enumerate(target_names):
    top10=np.argsort(clf.coef_[i])[-10:]

print("accuracy score: " + str(model.score(X_test, Y_test)))

print(model.predict(['hoihoi']))

