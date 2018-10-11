import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from DataEditer import DataEdit

data = pd.read_csv('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\resume_dataset.csv')
resumes = data["Resume"].values
labels = data["Category"].values


resumes = DataEdit.stringRemoverAndStopword(resumes)
resumes = DataEdit.stemAndStopText(resumes)


X_train, X_test, Y_train, Y_test = train_test_split(resumes, labels, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=100000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

model = pipeline.fit(X_train, Y_train)

print(X_train, Y_train)
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

print(model.predict(['java informatics software developer']))

