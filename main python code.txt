import numpy as np
import pandas as pd
df=pd.read_csv("19.csv")
p=df['ph'].values
l=df["ldr"].values
r=df["result"].values
X = list(zip(p,l))
Y = list(r)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[12.4, 417]]))
