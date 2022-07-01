import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import pickle

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    
#checking the data shape for the training data
print(X_credit_treinamento.shape, y_credit_treinamento.shape)

#checking the data shape for the test data
print(X_credit_teste.shape, y_credit_teste.shape)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)

#comparing the values betwen the traineing and test results
from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy %
print(accuracy_score(y_credit_teste, previsoes))

confusion_matrix(y_credit_teste, previsoes)

#to visualize
'''
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
'''
print(classification_report(y_credit_teste, previsoes))
