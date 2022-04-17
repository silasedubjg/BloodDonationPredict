from turtle import shape
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression

import csv

ficheiro = open('BloodDonation.csv','rb')
colunas = ["Recência", "Frequência",
           "Total doado(ml)", "Tempo(meses)", "Potencial Doador"]
dataset = pd.read_csv(ficheiro, names=colunas, skiprows=1, delimiter=";")

print(dataset.shape)

print(dataset.head())
