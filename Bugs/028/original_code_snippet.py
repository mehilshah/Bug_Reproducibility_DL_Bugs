df = pd.read_csv("adult_data.csv",header=None)
X = df.iloc[:,0:14]
Y = df.iloc[:,14]

encoder = LabelEncoder()
#X
for i in [1,3,5,6,7,8,9,13]:
   column = X[i]
   encoder.fit(column)
   encoded_C = encoder.transform(column)
   X[i] = np_utils.to_categorical(encoded_C)

print(X.shape)
#Y
encoder.fit(Y)
en_Y = encoder.transform(Y)
Y = np_utils.to_categorical(en_Y)

#model
model = Sequential()
model.add(Dense(21, input_dim=14, activation="relu"))
model.add(Dense(2, activation="softmax"))
#compile
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=
["accuracy"])

#train
model.fit(X,Y, epochs=50, batch_size=100)
score = model.evaluate(X,Y)
print("Accuracy: {}%".format(score[0]))