import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

base_risco_credito = pd.read_csv('risco_credito.csv')

X_risco_credito = base_risco_credito.iloc[:, 0:4].values 

y_risco_credito = base_risco_credito.iloc[:, 4].values

#transforming the category data in numerical data
from sklearn.preprocessing import LabelEncoder

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantias.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

import pickle
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

#naive-bayes algorith
naive_risco_credito = GaussianNB()
#training the algorith
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

#new client to preview:
#história boa (0), dívida alta(0), garantias nenhuma(1), renda > 35(2)
#história ruim(2), dívida alta(0), garantias adequada(0), renda < 15(0)

#generating the preview
previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
print('Para os dois clientes novos, seus respectivos riscos de concessão de empréstimo serão : ', previsao)

#analise the counts:
naive_risco_credito.class_count_
naive_risco_credito.classes_
naive_risco_credito.class_prior_ 

######################################################################################
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    
#checking the data shape for the training data
print(X_credit_treinamento.shape, y_credit_treinamento.shape)

#checking the data shape for the test data
print(X_credit_teste.shape, y_credit_teste.shape)

naive_creit_data = GaussianNB
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

