# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:48:28 2019

@author: Paatryk
"""
# import bibliotek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import BazDanych
dataset1 = pd.read_csv('season-0910.csv')
dataset2 = pd.read_csv('season-1011.csv')
dataset3 = pd.read_csv('season-1112.csv')
dataset4 = pd.read_csv('season-1213.csv')
dataset5 = pd.read_csv('season-1314.csv')
dataset6 = pd.read_csv('season-1415.csv')
dataset7 = pd.read_csv('season-1516.csv')
dataset8 = pd.read_csv('season-1617.csv')
dataset9 = pd.read_csv('season-1718.csv')
dataset10 =pd.read_csv('season-1819.csv')

y = dataset10.iloc[:, 5].values
x = dataset10.iloc[:, [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values

#zamiana tzw categorical data na 0 badz 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 2] = labelencoder_x.fit_transform(x[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)
# podzielenie na dane treningowe oraz dane testowe
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
