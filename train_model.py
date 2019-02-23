import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers import MaxPooling2D
import os

data = "data/"
X = []
y = []

for file in os.listdir(data):
    class_name = file.split('_')[3][:-4]
    class_samples = np.load("data/" + file)
    for s in class_samples:
        X.append(np.reshape(s, (28, 28)))
        y.append(class_name)

encoder = LabelBinarizer()
y_categ = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_categ, test_size = 0.2, shuffle=True, stratify=y_categ)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'