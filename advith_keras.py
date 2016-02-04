from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

# CONSTANTS
TRAINING = 3351
EPOCHS = 180
losses = ["mse", "rmse", "mae", "mape", "msle", "squared_hinge", "hinge", "binary_crossentropy"]
optimizers = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax"]
activations = ["tanh", "sigmoid", "hard_sigmoid"]
inits = ["uniform", "lecun_uniform", "normal", "zero", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]
data = np.loadtxt("training_data.txt", delimiter="|", skiprows=1)

allX = data[:, 0:-1]
allY = data[:, -1]

trainingX = data[0:TRAINING, 0:-1]
trainingY = data[0:TRAINING, -1]

testX = data[TRAINING:, 0:-1]
testY = data[TRAINING:, -1]

test_data = np.loadtxt("testing_data.txt", delimiter="|", skiprows=1)

def score(res):
    score = 0
    for x in range(len(res)):
        if (res[x] < .5):
            temp = 0
        else:
            temp = 1

        if (temp == testY[x]):
            score += 1
    return score/float(838)

def modele(l,o,a,i):
    model = Sequential()
    model.add(Dense(1, input_dim=1000, init=i, activation=a))
    model.compile(loss=l,
              optimizer=o,
              class_mode='binary')
    model.fit(trainingX, trainingY,
          nb_epoch=EPOCHS,
          batch_size=256,
          show_accuracy=True, verbose=0)
    res = model.predict(testX, batch_size=5)
    return score(res)
'''
opt = ("loss", "optimizer", "activation", "init")
maxVal = 0
count = 0
for l in losses:
    for o in optimizers:
        for a in activations:
            for i in inits:
                count += 1
                s1 = model(l, o, a, i)
                s2 = model(l, o, a, i)
                s3 = model(l, o, a, i)
                res = (s1 + s2 + s3)/3.0
                print "Number : " + str(count) + "  |  " + l + " " + o + " " + a + " " + i + " : " + str(res)
                if (res > maxVal):
                    maxVal = res
                    opt = (l, o, a, i)

print "---------------------------------------------------------"
print maxVal
print opt
'''
model = Sequential()
model.add(Dense(1, input_dim=1000, init='he_uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy',
          optimizer='adagrad',
          class_mode='binary')
model.fit(trainingX, trainingY,
      nb_epoch=EPOCHS,
      batch_size=256,
      show_accuracy=True, verbose=0)
res = model.predict(test_data, batch_size=5)

predictions = []
for x in range(len(res)):
    if (res[x] < .5):
        predictions.append(0)
    else:
        predictions.append(1)


f = open("keras.csv", "w")
f.write("Id,Prediction\n")

for x in range(len(predictions)):
    f.write(str(x+1) + "," + str(predictions[x]) + "\n")

f.close()

