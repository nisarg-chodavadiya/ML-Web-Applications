import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pickle

df = pd.read_csv('concrete_data.csv')

X = df[['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer','coarse_aggregate', 'fine_aggregate', 'age']]
y = df['concrete_compressive_strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

regr = RandomForestRegressor(n_estimators=5, random_state=0)
regr.fit(X_train, y_train)

pickle.dump(regr, open('concrete.pkl', 'wb'))
