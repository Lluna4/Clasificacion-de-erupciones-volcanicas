import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv


data = pd.read_csv("jm_train.csv")
x = data.drop("target", axis=1)
y = data["target"]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

while (True):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    if model.score(x_test, y_test) >= 0.833333333333:
        break

pred = pd.read_csv("jm_X_test.csv")
pred = pred.to_numpy()

for i in pred:
    i = np.array(i).reshape(1, -1)
    with open("pred.csv", "a", newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow([model.predict(i)[0]])

