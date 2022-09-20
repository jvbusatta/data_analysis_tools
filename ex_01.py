from email.mime import base
from statistics import mean
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv('D:\PYTHON\AULAS\curso_python_a_z\data_analysis_tools\credit_data.csv')

print(base_credit.head(10))
print(base_credit.describe())
print('------------------------------------------')
print(base_credit[base_credit['income'] >= 69995])
print('------------------------------------------')
print(base_credit[base_credit['loan'] >= 1.3776])
print('------------------------------------------')
print(np.unique(base_credit['default'], return_counts=True))
#1717 customers who pay a loan and 283 who do not pay a loan

sns.countplot(x = base_credit['default']) 

plt.hist(x = base_credit['age'])
plt.hist(x = base_credit['income']) 
plt.hist(x = base_credit['loan'])

#scatter chart for data evaluation
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()

#correction of data values
print(base_credit.loc[base_credit['age'] < 0 ])
#outra maneira
print(base_credit[base_credit['age'] < 0])
#1 - delete the entire column (from all records in the database)
base_credit2 = base_credit.drop('age', axis=1)
print(base_credit2) 

#delete only the records with inconsistent values - only the 'ages' < 0
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit3)
print(base_credit3.loc[base_credit3['age']<0])

#recommended mode
#manually fill in customer data
#fill in with the average age
print(base_credit.mean())
print(base_credit['age'].mean())

base_credit['age'][base_credit['age']>0].mean()

#replace outlier values with the average from the existing database
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.927
print(base_credit.head(27))
#---------------------------------------------------------------------
#for handling missing values
print(base_credit.isnull())
#best visualization
print(base_credit.isnull().sum())
#to find the null values for the missing data
print(base_credit.loc[pd.isnull(base_credit['age'])])
#to fill the null values
print(base_credit['age'].fillna(base_credit['age'].mean(), inplace = True))
#to find the null values for the missing data
print(base_credit.loc[pd.isnull(base_credit['age'])])
#locating the other missing values
print(base_credit.loc[pd.isnull(base_credit['clientid'])])
#locating specifics data 
print(base_credit.loc[pd.isnull(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)])
#other easier method to locate
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

#---------------------------------------------------------------------
#dividing the variables for previsor(x) and for classes(y)
X_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

#---------------------------------------------------------------------
#visualizing the extreme points of variables
print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

#the values need to be standardized to applie machine learning methods

#standardisation - outliers values

from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
#now, the machine learning methods do not make mistakes with outliers values, they are in the same scale

#normalization - 

#*********************************************************************
#base_census analizing
base_census = pd.read_csv('census.csv')
#analyzing data
base_census.describe()
base_census.isnull().sum()
#objectively analyzing the data
np.unique(base_census['income'], return_counts=True)
#ploting the data
sns.countplot(x = base_census['income'])
#histogram for 'age' attribute
plt.hist(x = base_census['age'])
#another histogram for 'education-num' attribute
plt.hist(x = base_census['education-num'])
#another histogram for 'hour-per-week' attribute
plt.hist(x = base_census['hour-per-week'])
#generating dynamic graphics
grafico = px.treemap(base_census, path = ['workclass', 'age'])
grafico.show()

grafico1 = px.treemap(base_census, path = ['occupation', 'relationship'])
grafico1.show()

grafico3 = px.parallel_categories(base_census, dimensions = ['occupation', 'relationship'])
grafico3.show()

grafico3 = px.parallel_categories(base_census, dimensions = ['workclass','occupation', 'income'])
grafico3.show()

grafico4 = px.parallel_categories(base_census, dimensions = ['education', 'income'])
grafico4.show()


X_census = base_census.iloc[:, 0:14].values
base_census.columns

y_census = base_census.iloc[:, 14].values

#transforming string classification in numbers
from sklearn.preprocessing import LabelEncoder

label_encoder_teste = LabelEncoder()
teste = label_encoder_teste.fit_transform(X_census[:, 1])
#creating a variable ofr each class
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

#making all the attributes have the same importance
X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(X_census.shape)

#appling the normalization on data
from sklearn.preprocessing import StandardScaler

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

#divinding the traing and test data bases
from sklearn.model_selection import train_test_split

#for credit database
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size = 0.25, random_state = 0)
#evaluating the data len
print(X_credit_treinamento.shape)
print(y_credit_treinamento.shape)
print(X_credit_teste.shape, y_credit_teste.shape) 

#for census database
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_credit, y_credit, test_size = 0.25, random_state = 0)
print(X_census_treinamento.shape)
print(y_census_treinamento.shape)
print(X_census_teste.shape, y_census_teste.shape) 

#saving the database
import pickle

with open('credit.pkl', mode = 'wb') as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste], f)

with open('census.pkl', mode = 'wb') as f:
    pickle.dump([X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste, f])
    
#writining the code