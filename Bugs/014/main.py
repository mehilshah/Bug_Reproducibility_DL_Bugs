from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import time

X_3 = np.random.rand(100, 1, 20, 56)
y_3 = np.random.randint(0, 2, size=100)
y_3 = np.eye(2)[y_3]

model = Sequential()
model.add(Conv2D(32, (3, 3),activation='relu',input_shape=(1, 20, 56)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3),  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
start = time.time()
model_info = model.fit(X_3, y_3, batch_size=100, \
                         epochs=20, verbose=2)
end = time.time()