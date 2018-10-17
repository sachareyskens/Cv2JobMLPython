import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import DataEditer
from DataEditer import DataEdit


def run():
    X_train, X_test, Y_train, Y_test, labels = processData()
    model = buildPipeline(X_train, Y_train)
    trainedModel = train(model, labels, X_test, Y_test)
    predict(trainedModel, ['hi software java development etc informatics'])

def buildPipeline(X_train, Y_train):
    pipeline = Pipeline(
        [('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True, min_df=1, use_idf=True)),
         ('chi', SelectKBest(chi2, k=60000)),
         ('clf', SGDClassifier(penalty='l1', learning_rate='optimal', eta0=0.1, verbose=1, n_iter=300, n_jobs=40))])
    model = pipeline.fit(X_train, Y_train)
    return model

def train(model, labels, X_test, Y_test):
    clf = model.named_steps['clf']
    target_names = set(labels)
    for i, label in enumerate(target_names):
        top10 = np.argsort(clf.coef_[i])[-10:]
    print("accuracy score: " + str(model.score(X_test, Y_test)))
    return model

def processData():
    DataEditer.setInputFile('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\resume_dataset.csv')
    data = DataEditer.readData()
    resumes = data["Resume"].values
    labels = data["Category"].values
    resumes = DataEdit.stringRemoverAndStopword(resumes)
    resumes = DataEdit.stemAndStopText(resumes)
    X_train, X_test, Y_train, Y_test = train_test_split(resumes, labels, test_size=0.2)
    return X_train, X_test, Y_train, Y_test, labels

def predict(model, toPredict):
    print(model.predict(toPredict))


run()