import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

import pickle

with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

print(X_census_treinamento.shape, y_census_treinamento.shape)
print(X_census_teste.shape, y_census_teste.shape)

naive_census = GaussianNB()

naive_census.fit(X_census_treinamento, y_census_treinamento)
previsoes = naive_census.predict(X_census_teste)
print(previsoes)
#comparing the predict results with the real results
print(y_census_teste)

print('Taxa de acerto: ', accuracy_score(y_census_teste, previsoes))

cm = confusion_matrix(naive_census, y_census_teste)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))