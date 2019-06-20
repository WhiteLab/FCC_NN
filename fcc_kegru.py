import sys
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, TimeDistributed, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping


try:
    data_path = sys.argv[1]
except:
    data_path = 'rnn_data.h5'

import h5py
import numpy as np
f = h5py.File(data_path)
X_train = np.array(f['X_train'])
X_test = np.array(f['X_test'])
X_valid = np.array(f['X_valid'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])
y_valid = np.array(f['y_valid'])

y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))


# print(X_train[0])
# print(y_train[0])
# print(X_train[0].shape)
# print(y_train[0].shape)

model = Sequential()
model.add(Bidirectional(GRU(20, return_sequences=True), batch_input_shape=(None, None, 50)))
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mae'])


model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=128)
print('Test accuracy: {}'.format(model.evaluate(X_test, y_test)))
