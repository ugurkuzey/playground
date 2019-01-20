# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:43:18 2019

@author: Ugur
"""

import pandas as pd
import matplotlib as plt
from pandas.plotting import andrews_curves
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 6)

iris = pd.read_csv("Iris.csv")
print(iris.head())
print(iris["Species"].value_counts())
iris.info()
iris.drop("Id", axis=1, inplace=True)

fig = iris[iris.Species == "Iris-setosa"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", color="red", label="setosa")
iris[iris.Species == "Iris-versicolor"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", color="green", label="versicolor", ax=fig)
iris[iris.Species == "Iris-virginica"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", color="blue", label="virginica", ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
#plt.show()

fig = iris[iris.Species == "Iris-setosa"].plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm", color="red", label="setosa")
iris[iris.Species == "Iris-versicolor"].plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm", color="green", label="versicolor", ax=fig)
iris[iris.Species == "Iris-virginica"].plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm", color="blue", label="virginica", ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
#plt.show()

print(iris.corr())

#andrews_curves(iris, "Species")

#x = iris.iloc[:,:4].values
x = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
print(x.head())

#y = iris.iloc[:,4:].values
y = iris["Species"]
print(y.head())

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print("Logistic Regression")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# KNN - K-Nearest Neighbours
knn_model  = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
print("KNN - K-Nearest Neighbours")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SVM - Support Vector Machine
svm_model = SVC()
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
print("SVM - Support Vector Machine")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred = nb_model.predict(x_test)
print("Naive Bayes")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_pred = dt_model.predict(x_test)
print("Decision Tree")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Random Forest
rf_model = RandomForestClassifier(max_depth=3)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
print("Random Forest")
print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))