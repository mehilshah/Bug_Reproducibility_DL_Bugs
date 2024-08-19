from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense, Activation

data = pd.read_csv('text_class.csv')
train_text = data['query']
train_labels = data['tags']

tokenize = Tokenizer(num_words=100)
tokenize.fit_on_texts(train_text)

x_data = tokenize.texts_to_matrix(train_text)

encoder = LabelBinarizer()
encoder.fit(train_labels)
y_data = encoder.transform(train_labels)

model = Sequential()
model.add(Dense(512, input_shape=(100,)))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=8, epochs=10)

predictions = model.predict(x_data[0])
tag_labels = encoder.classes_
predicted_tags = tag_labels[np.argmax(predictions)]
print (predicted_tags)