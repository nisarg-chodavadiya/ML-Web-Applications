import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('iris.csv')

X = np.array(df.iloc[:, :4])
y = np.array(df.iloc[:, 4:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

clf = RandomForestClassifier(n_estimators=3, random_state=1)
clf = clf.fit(X_train, y_train)

pickle.dump(clf, open('iris.pkl', 'wb'))
