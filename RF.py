randfor_model1 = RandomForestClassifier(n_estimators=10000000, min_samples_leaf=5, max_features=62, n_jobs=-1)
randfor_model1.fit(dataX, dataY)

# Predict data and write to file.
randfor_predict1 = randfor_model1.predict(test_data)

f = open("RandomForestJoon5.csv", "w")
f.write("Id,Prediction\n")
for x in range(len(randfor_predict1)):
    f.write(str(x+1) + "," + str(int(randfor_predict1[x])) + "\n")
f.close()

print "DONE"