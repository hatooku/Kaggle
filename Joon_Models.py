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
# Data Playground!
all_dataX = np.vstack((dataX, test_data))

# Variance Threshold - KEEP
from sklearn.feature_selection import VarianceThreshold

varThreshold = VarianceThreshold()

all_dataX = varThreshold.fit_transform(all_dataX)
dataX = all_dataX[:4189, :]

test_data = all_dataX[4189:, :]

#Tf-Idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

tfidf.fit(all_dataX)
all_dataX_scaled = tfidf.transform(all_dataX)

dataX_scaled = all_dataX_scaled[:4189, :]
trainingX_scaled = dataX_scaled[0:training_size]
testX_scaled = dataX_scaled[training_size:]

test_data_scaled = all_dataX_scaled[4189:, :]

####################################################################
####################################################################
print "start"

#85 mins!
rf_sub = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et_sub = ExtraTreesClassifier(n_estimators=50000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg_sub = XGBClassifier(n_estimators=20000, learning_rate=0.1, colsample_bytree=0.51, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)

vote1 = VotingClassifier(estimators=[('rf', rf_sub), ('et', et_sub), ('xg', xg_sub), ('ada', ada_sub)], weights=[1, 1, 1, 2], voting='hard')


start_time = time.time()
vote1.fit(dataX_scaled, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))


start_time = time.time()
vote1_predict = vote1.predict(test_data_scaled)
f = open("Ensemble3.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote1_predict)):
    f.write(str(x+1) + "," + str(int(vote1_predict[x])) + "\n")
f.close()
print("--- %.2f mins ---" % ((time.time() - start_time)/60))

os.system('say "your program has finished"')