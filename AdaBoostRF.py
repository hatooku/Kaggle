import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
import time
import os
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import datetime

####################################################################

# Load training and test data (80/20 Split)
data = np.loadtxt("training_data.txt", delimiter="|", skiprows=1)
dataX = data[:, 0:-1]
dataY = data[:, -1]
training_size = int(data.shape[0] * 0.8)

trainingX = data[0:training_size, 0:-1]
trainingY = data[0:training_size, -1]

testX = data[training_size:, 0:-1]
testY = data[training_size:, -1]

# For testing for submission

test_data = np.loadtxt("testing_data.txt", delimiter="|", skiprows=1)

####################################################################

# RF
start_time = time.time()
adarf_sub3 = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1, random_state=712)
ada_sub3 = AdaBoostClassifier(base_estimator=adarf_sub3, n_estimators=13, learning_rate=1, random_state=777)
ada_sub3.fit(dataX, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))

os.system('say "Master, your program has finished"')

# Predict data and write to file.
ada_sub3_predict = ada_sub3.predict(test_data)

f = open("AdaBoostRF7.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(ada_sub3_predict)):
    f.write(str(x+1) + "," + str(int(ada_sub3_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
datetime.datetime.now()