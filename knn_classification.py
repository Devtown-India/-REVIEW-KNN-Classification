# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder
#dataset path

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#column names
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#read dataset and show dataset
dataset = pd.read_csv(path, names = headernames)
print(dataset.head(100))

# in X get all input data and in y get output data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(y)

# labels = LabelEncoder()
# labels.fit(y)
# y_labels = labels.transform(y)
# print(y_labels)

#use train_test_split method to devide your data into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#use standard scalar normalization to get better accuracy
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# create knn model and fit train data
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)

# predict value from model
y_pred = classifier.predict(X_test)

#create confusion metrix get better understanding of output especially for classification problem
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

# get accuracy of our model
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

scalers = StandardScaler()
ar = [[5.1,3.5,1.4,0.2]]
# ar = scalers.fit_transform(ar)
# print(ar)
pre = classifier.predict(ar)

print(pre)
# pree = labels.inverse_transform(pre)
# print(pree)

# save our knn model
# pickle.dump(classifier,open('knn_model.pickle','wb'))  