# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)\
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data=pd.read_csv('C:\\Users\\sachare\\Documents\\Github\\Cv2JobMLPython\\input\\resume_dataset.csv')
print(data)

cvs = data['Resume'].values
cvs = dict(zip(cvs, range(len(cvs))))
data['Resume_encode'] = data['Resume'].replace(cvs)
print(data)

cvs_onehot = OneHotEncoder(sparse=False).fit_transform(data['Resume_encode'].values.reshape(-1,1))
print(cvs_onehot.shape)
data=data.join(pd.DataFrame(cvs_onehot, columns=cvs.keys()), how='outer')

data.drop(['Resume'], axis='columns', inplace=True)
print(data)

y, x = data['Category'].values, data.drop('Category', axis='columns').values
x_std = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.20)

dummy_classifier = DummyClassifier(strategy="prior")
dummy_classifier.fit( x_test, y_test )
print(dummy_classifier.score(x_test, y_test))