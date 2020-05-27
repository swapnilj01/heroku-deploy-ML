import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.under_sampling import NearMiss

df = pd.read_csv('creditcard.csv')
X = df.drop("Class",1)   #Feature Matrix
y = df["Class"]  

#Using Best features

X_small = X[['V10','V11','V12','V14','V16','V17']]

nm = NearMiss(sampling_strategy = {0:100000 , 1: 492})

X_res,y_res=nm.fit_sample(X_small,y)

clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=50)
clf.fit(X_res,y_res)

pickle.dump(clf,open('model.pkl','wb'))