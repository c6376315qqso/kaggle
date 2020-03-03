import keras
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.utils import to_categorical
import numpy as np
import matplot as mp

data_train = pd.read_csv(r'C:\Users\Jame Black\Desktop\train.csv')
X_test = pd.read_csv(r'C:\Users\Jame Black\Desktop\test.csv')
X_train = data_train.iloc[:, 1:]
Y_train = data_train.iloc[:, 0]
X_train /= 255
X_test /= 255
Y_train = to_categorical(Y_train, 10)

X_test = np.array(X_test)
X_test = X_test.reshape([X_test.shape[0], 28, 28, 1])

X_train = np.array(X_train)
X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])

model = Sequential([
    Conv2D(32, (7,7), strides=(1,1), activation='relu', input_shape=(28,28,1)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, (5,5), strides=(1,1), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs=2)
Y_test = model.predict_classes(X_test)
Y_test = pd.DataFrame(Y_test).astype(int)
Y_test.columns = ['Label']
Y_test.insert(0, column='ImageId', value=range(1,Y_test.shape[0] + 1))
Y_test.to_csv(r'C:\Users\Jame Black\Desktop\result.csv', index=False)
