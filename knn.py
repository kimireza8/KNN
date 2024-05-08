from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

dataset = pd.read_csv('milk.csv')
train_data = np.array(dataset)[:, :-1]
train_label = np.array(dataset)[:, -1]

print(train_data)
print(train_label)


sc = MinMaxScaler(feature_range=(0, 1))
data = sc.fit_transform(train_data)
print(data)

KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
KNN.fit(train_data, train_label)

ph = input('Tulis input ph : ')
temp = input('Tulis input temp : ')
taste = input('Tulis input taste : ')
odor = input('Tulis input odor : ')
fat = input('Tulis input fat : ')
turbid = input('Tulis input turbidp : ')
color = input('Tulis input color: ')

test_data = np.array([float(ph), int(temp), int(taste), int(odor), int(fat), int(turbid), int(color)])
test_data = np.reshape(test_data, (1,-1))
print(test_data)

hasil = KNN.predict(test_data)
print("hasil dari knn = " ,hasil)
