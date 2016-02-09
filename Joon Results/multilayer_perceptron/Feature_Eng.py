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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.svm import SVC, NuSVC, LinearSVC
from multilayer_perceptron  import MultilayerPerceptronClassifier

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
trainingX = dataX[0:training_size, :]
testX = dataX[training_size:, :]
test_data = all_dataX[4189:, :]

####################################################################


print "start: " + str(datetime.datetime.now())

# Logistic Regression

LR = LogisticRegression(n_jobs=-1)
LR_scores = cross_val_score(LR, dataX_scaled_sp, dataY, cv=10, n_jobs=-1)
print LR_scores
print np.average(LR_scores)


# Multinomial Naive Bayes
start_time = time.time()
MNB = MultinomialNB(alpha=0.0, fit_prior=False)

MNB_scores = cross_val_score(MNB, dataX_scaled_sp, dataY, cv=100, n_jobs=-1)

print MNB_scores
print np.average(MNB_scores)



#print("--- %.2f mins ---" % ((time.time() - start_time)/60))
#os.system('say "Master, your program has finished"')

#print MNB.score(testX_scaled, testY)

from sklearn.svm import SVC, NuSVC, LinearSVC
svc = LinearSVC(C=0.72)
svc.fit(trainingX_scaled, trainingY)
print svc.score(testX_scaled, testY)


#RF = RandomForestClassifier(n_estimators=1000, n_jobs=-1)


#Voter = VotingClassifier([('LR', LR), ('MNB', MNB)], voting='hard')


#MNB.fit(dataX_scaled, dataY)
# Predict data and write to file.
#MNB_predict = MNB.predict(test_data_scaled)

#f = open("MultiNB1.csv", "w")
#f.write("Id,Prediction\n")
#for x in range(len(MNB_predict)):
#    f.write(str(x+1) + "," + str(int(MNB_predict[x])) + "\n")
#f.close()

#os.system('say "Master, your file has been created."')
#print datetime.datetime.now()






rf = RandomForestClassifier(n_estimators=300000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=150000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg = XGBClassifier(n_estimators=60000, learning_rate=0.1, colsample_bytree=0.51007979, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=30000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)

vote = VotingClassifier([('rf', rf), ('et', et), ('xg', xg), ('ada', ada_sub)], voting='hard', weights=[1,1,1,2])

start_time = time.time()

vote.fit(dataX, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))

os.system('say "Master, your program has finished"')

# Predict data and write to file.
vote_predict = vote.predict(test_data)

f = open("Ensemble3.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote_predict)):
    f.write(str(x+1) + "," + str(int(vote_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
datetime.datetime.now()


#0.76014, CV: 0.712590355255
start_time = time.time()
rf = RandomForestClassifier(n_estimators=1000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
rf_scores = cross_val_score(rf, dataX_scaled, dataY, cv=10, n_jobs=-1)
print rf_scores
print np.average(rf_scores)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))


import random
rf_arr = []

#Try: Default, random parameters, increase number of, increase estimators
#Default:
    #100 x 100: 0.75656
    #10 x 100: 0.754
    #10 x 1000: 0.754
    # random x 10: 0.748
    # random x 100: 0.747

# estimated time: 2 mins
for i in range(100):
    # Individual Score: 0.74343675417661093
    rf = RandomForestClassifier(n_estimators=100, max_features=random.randint(50, 80), criterion='entropy', min_samples_leaf=random.randint(1, 6), n_jobs=-1)
    rf_tuple = (str(i), rf)
    rf_arr.append(rf_tuple)
    
vote = VotingClassifier(rf_arr, voting='hard')

start_time = time.time()
vote_score = cross_val_score(vote, dataX_scaled, dataY, cv=10, n_jobs=-1)
print vote_score
print np.average(vote_score)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))

#100:   0.743
#1000:  0.7604143198, 0.7613365
#10000: 0.7577565



################################################

# Hyperparameter Tuning

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
param_dist = {"max_features": sp_randint(15, 80),
              "min_samples_leaf": sp_randint(1, 11),
              "criterion": ["gini", "entropy"]}
n_iter_search = 60
rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=n_iter_search, cv = 5, n_jobs=-1)

start_time = time.time()
rf_grid.fit(dataX_scaled, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print rf_grid.best_score_
print rf_grid.best_params_
os.system('say "your first program has finished"')

et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
et_grid = RandomizedSearchCV(estimator=et, param_distributions=param_dist, n_iter=n_iter_search, cv = 5, n_jobs=-1)

start_time = time.time()
et_grid.fit(dataX_scaled, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print et_grid.best_score_
print et_grid.best_params_
os.system('say "your second program has finished"')




#####
print datetime.datetime.now()
adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=26, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)

start_time = time.time()

ada_sub.fit(dataX_scaled, dataY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))

os.system('say "Master, your program has finished"')

# Predict data and write to file.
ada_predict = ada_sub.predict(test_data_scaled)

f = open("AdaBoostRF8.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(ada_predict)):
    f.write(str(x+1) + "," + str(int(ada_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print datetime.datetime.now()


####




# Create MLP Object
# Please see line 562 in "multilayer_perceptron.py" for more information 
# about the parameters
#mlp = MultilayerPerceptronClassifier(hidden_layer_sizes = (50, 20), max_iter = 200, alpha = 0.02)


#0.7517
start_time = time.time()
mlp = MultilayerPerceptronClassifier(hidden_layer_sizes = (300, 200, 70, 50), algorithm='sgd', batch_size=200, max_iter=200, shuffle=True, tol=1e-5, learning_rate_init=0.5)

ada = AdaBoostClassifier(mlp, n_estimators=20)

ada.fit(trainingX_scaled, trainingY)
print ada.score(testX_scaled, testY)
print("--- %.2f mins ---" % ((time.time() - start_time)/60))




start_time = time.time()
MNB = ExtraTreesClassifier(n_jobs=-1, max_features=49, criterion='gini', min_samples_leaf=2, n_estimators=40000)
ada = AdaBoostClassifier(MNB, n_estimators=40)
ada.fit(dataX, dataY)

ada_predict = ada.predict(test_data)

f = open("AdaET2.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(ada_predict)):
    f.write(str(x+1) + "," + str(int(ada_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()




#0.757 (40)








# SUPER ENSEMBLE 4
rf = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=50000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg = XGBClassifier(n_estimators=20000, learning_rate=0.1, colsample_bytree=0.51007979, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)


LR = LogisticRegression(n_jobs=-1)
MNB = MultinomialNB(alpha=0.0, fit_prior=False)
SVC = LinearSVC(C=0.72)
MLP = MultilayerPerceptronClassifier(hidden_layer_sizes = (300, 200, 70, 50), algorithm='sgd', batch_size=200, max_iter=200, shuffle=True, tol=1e-5, learning_rate_init=0.5)

voting_arr = [('rf', rf), ('et', et), ('xg', xg), ('ada', ada_sub), ('LR', LR), ('MNB', MNB), ('SVC', SVC), ('MLP', MLP)]
weights = [1, 1, 1, 2, 1, 1, 1, 1]

vote = VotingClassifier(voting_arr, weights=weights, voting='hard')

start_time = time.time()
vote.fit(dataX, dataY)
vote_predict = vote.predict(test_data)

f = open("SuperEnsemble4.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote_predict)):
    f.write(str(x+1) + "," + str(int(vote_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()

# SUPER ENSEMBLE 5
rf = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=50000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg = XGBClassifier(n_estimators=20000, learning_rate=0.1, colsample_bytree=0.51007979, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)


LR = LogisticRegression(n_jobs=-1)
MNB = MultinomialNB(alpha=0.0, fit_prior=False)
SVC = LinearSVC(C=0.72)
MLP = MultilayerPerceptronClassifier(hidden_layer_sizes = (300, 200, 70, 50), algorithm='sgd', batch_size=200, max_iter=200, shuffle=True, tol=1e-5, learning_rate_init=0.5)

voting_arr = [('rf', rf), ('et', et), ('xg', xg), ('ada', ada_sub), ('LR', LR), ('MNB', MNB), ('SVC', SVC), ('MLP', MLP)]
weights = [1, 1, 1, 2, 1, 1, 1, 1]

vote = VotingClassifier(voting_arr, weights=weights, voting='hard')

start_time = time.time()
vote.fit(dataX, dataY)
vote_predict = vote.predict(test_data)

f = open("SuperEnsemble5.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote_predict)):
    f.write(str(x+1) + "," + str(int(vote_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()

# SUPER ENSEMBLE 6
rf = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=50000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg = XGBClassifier(n_estimators=20000, learning_rate=0.1, colsample_bytree=0.51007979, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)


LR = LogisticRegression(n_jobs=-1)
MNB = MultinomialNB(alpha=0.0, fit_prior=False)
SVC = LinearSVC(C=0.72)
MLP = MultilayerPerceptronClassifier(hidden_layer_sizes = (300, 200, 70, 50), algorithm='sgd', batch_size=200, max_iter=200, shuffle=True, tol=1e-5, learning_rate_init=0.5)

voting_arr = [('rf', rf), ('et', et), ('xg', xg), ('ada', ada_sub), ('LR', LR), ('MNB', MNB), ('SVC', SVC), ('MLP', MLP)]
weights = [1, 1, 1, 2, 1, 1, 1, 1]

vote = VotingClassifier(voting_arr, weights=weights, voting='hard')

start_time = time.time()
vote.fit(dataX, dataY)
vote_predict = vote.predict(test_data)

f = open("SuperEnsemble6.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote_predict)):
    f.write(str(x+1) + "," + str(int(vote_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()

# SUPER ENSEMBLE 7
rf = RandomForestClassifier(n_estimators=100000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=50000, max_features=81, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
xg = XGBClassifier(n_estimators=20000, learning_rate=0.1, colsample_bytree=0.51007979, max_depth=7, min_child_weight=2)

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada_sub = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)


LR = LogisticRegression(n_jobs=-1)
MNB = MultinomialNB(alpha=0.0, fit_prior=False)
SVC = LinearSVC(C=0.72)
MLP = MultilayerPerceptronClassifier(hidden_layer_sizes = (300, 200, 70, 50), algorithm='sgd', batch_size=200, max_iter=200, shuffle=True, tol=1e-5, learning_rate_init=0.5)

voting_arr = [('rf', rf), ('et', et), ('xg', xg), ('ada', ada_sub), ('LR', LR), ('MNB', MNB), ('SVC', SVC), ('MLP', MLP)]
weights = [1, 1, 1, 2, 1, 1, 1, 1]

vote = VotingClassifier(voting_arr, weights=weights, voting='hard')

start_time = time.time()
vote.fit(dataX, dataY)
vote_predict = vote.predict(test_data)

f = open("SuperEnsemble7.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(vote_predict)):
    f.write(str(x+1) + "," + str(int(vote_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()

# AdaBoost

adarf_sub = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada = AdaBoostClassifier(base_estimator=adarf_sub, n_estimators=13, learning_rate=0.8)

start_time = time.time()
ada.fit(dataX, dataY)
ada_predict = ada.predict(test_data)

f = open("AdaRepeat.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(ada_predict)):
    f.write(str(x+1) + "," + str(int(ada_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()

# AdaBoost 2

adarf_sub2 = RandomForestClassifier(n_estimators=10000, max_features=65, criterion='entropy', min_samples_leaf=2, n_jobs=-1)
ada2 = AdaBoostClassifier(base_estimator=adarf_sub2, n_estimators=13, learning_rate=0.8)

start_time = time.time()
ada2.fit(dataX, dataY)
ada2_predict = ada2.predict(test_data)

f = open("AdaRepeat2.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(ada2_predict)):
    f.write(str(x+1) + "," + str(int(ada2_predict[x])) + "\n")
f.close()

os.system('say "Master, your file has been created."')
print("--- %.2f mins ---" % ((time.time() - start_time)/60))
print datetime.datetime.now()
