from email.mime import base
from statistics import mean
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import math

#calculating entrophy(Es)
#class_info = [Risk, Credit History High, Credit History Moderate, Credit History Low]
class_1 = [6/14, 1/5, 1/5, 3/5] #high
class_2 = [3/14, 2/5, 1/5, 2/5] #moderate
class_3 = [5/14, 3/4, 1/4, 0] #low


Es_risk = -class_1[0] * math.log(class_1[0],2) - class_2[0] * math.log(class_2[0],2) - class_3[0] * math.log(class_3[0],2)
Es_Credit_History_High = -class_1[1] * math.log(class_1[1],2) - class_2[1] * math.log(class_2[1],2) - class_3[1] * math.log(class_3[1],2)

print("Calculated Risk Entrophy : ", Es_risk)
print("Calculated Credit History High Entrophy : ", Es_Credit_History_High)
#credit history


#information gain(G)
#G(history)
#G(debt)
#G(guarantees)
#G(income)

G_history = 0.26




