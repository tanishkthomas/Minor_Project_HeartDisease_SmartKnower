import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


df = pd.read_csv('heart.csv')

X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]

# K-NN Classifier
BA1 = BestAccuracy()
BA1._init_(KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski'), X, Y, 100)
BA1.Accuracy()

# Support Vector Classifier
BA2 = BestAccuracy()
BA2._init_(SVC(kernel='linear', random_state=0), X, Y, 100)
BA2.Accuracy()

# Random Forest Classifer
BA3 = BestAccuracy()
BA3._init_(RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0), X, Y, 100)
BA3.Accuracy()
