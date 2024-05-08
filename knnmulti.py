from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

dataset = pd.read_csv('milk_training.csv')
train_data = np.array(dataset)[:, :-1]
train_label = np.array(dataset)[:, -1]
print("Train data:", train_data)
print("Train label:", train_label)

sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit_transform(train_data)
print("Data pembanding:", train_data_scaled)

datatest = pd.read_csv('milk_testing.csv')
test_data = np.array(datatest)[:, :-1]
test_label = np.array(datatest)[:, -1]
print("Test data:", test_data)
print("test label:", test_label)

test_data_scaled = sc.transform(test_data)
print("Yang diuji:", test_data_scaled)

KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
KNN.fit(train_data_scaled, train_label)

hasil = KNN.predict(test_data_scaled)
print("Hasil dari KNN:", hasil)
