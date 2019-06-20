import os
import sys
import argparse
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Convolution1D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

snppet_net = Sequential()
snppet_net.add(Convolution1D(
    name='convolution1d_10',
    batch_input_shape=[None, 50, 4],
    filters=120,  # nb_filter
    kernel_size=5,  # filter_length
    activation='linear'
))
snppet_net.add(Activation(name='activation_10', activation='relu'))
snppet_net.add(BatchNormalization(name='batchnormalization_10', epsilon=1e-3))
snppet_net.add(Dropout(name='dropout_10', rate=0.1))
snppet_net.add(Convolution1D(
    name='convolution1d_11',
    filters=120,
    kernel_size=5,
    activation='linear'
))
snppet_net.add(Activation(name='activation_11', activation='relu'))
snppet_net.add(BatchNormalization(name='batchnormalization_11', epsilon=1e-3))
snppet_net.add(Dropout(name='dropout_11', rate=0.1))
snppet_net.add(Convolution1D(
    name='convolution1d_12',
    filters=120,
    kernel_size=5,
    activation='linear'
))
snppet_net.add(BatchNormalization(name='batchnormalization_12', epsilon=1e-3))
snppet_net.add(Activation(name='activation_12', activation='relu'))
snppet_net.add(Dropout(name='dropout_12', rate=0.1))
snppet_net.add(Flatten(name='flatten_4'))
snppet_net.add(Dense(name='dense_4', units=2, activation='softmax'))

# Compile with ADAM optimizer, mean squared error loss function
snppet_net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load input and output data
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/cephfs/users/dominic/projects/EnhancerSeq/snppet_cnn/dataset.hdf5')
parser.add_argument('--subsample-rate', type=float, default=1.0)
parser.add_argument('--output-stub', default='encode_cnn')
args = vars(parser.parse_args())

enhancer_data = h5py.File(args['data'])
X_train = np.array(enhancer_data['X_train'])
y_train = np.array(enhancer_data['y_train'])
X_valid = np.array(enhancer_data['X_valid'])
y_valid = np.array(enhancer_data['y_valid'])
X_test = np.array(enhancer_data['X_test'])
y_test = np.array(enhancer_data['y_test'])

# Subsample input data, for testing purposes
"""train_subidx = np.random.choice(X_train.shape[0], int(X_train.shape[0] * args['subsample_rate']), replace=False)
valid_subidx = np.random.choice(X_valid.shape[0], int(X_valid.shape[0] * args['subsample_rate']), replace=False)
test_subidx = np.random.choice(X_test.shape[0], int(X_test.shape[0] * args['subsample_rate']), replace=False)
X_train = X_train[train_subidx, :]
y_train = y_train[train_subidx, :]
X_valid = X_valid[valid_subidx, :]
y_valid = y_valid[valid_subidx, :]
X_test = X_test[test_subidx, :]
y_test = y_test[test_subidx, :]
"""
# Run the model
snppet_net.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,
               callbacks=[EarlyStopping()])
metrics = snppet_net.evaluate(X_test, y_test, batch_size=128)
print('Test metrics: {}'.format(metrics))

# Save the model weights
snppet_net.save(args['output_stub'] + '_fullmodel.h5')
with open(args['output_stub'] + '_model.json', 'w') as jsonout:
    jsonout.write(snppet_net.to_json())
snppet_net.save_weights(args['output_stub'] + '_modelweights.h5')

"""
{'class_name': 'Sequential',
 'keras_version': '1.2.2',
 'config': [{'class_name': 'Convolution1D',
   'config': {'batch_input_shape': [None, 145, 4], *
    'W_constraint': None,
    'b_constraint': None,
    'name': 'convolution1d_10', *
    'activity_regularizer': None,
    'trainable': True, * 
    'filter_length': 5, * 
    'init': 'glorot_uniform',
    'bias': True,
    'nb_filter': 120,
    'input_dtype': 'float32',
    'subsample_length': 1,
    'border_mode': 'valid',
    'input_dim': None,
    'b_regularizer': None,
    'W_regularizer': None,
    'activation': 'linear',
    'input_length': None}},
  {'class_name': 'Activation',
   'config': {'activation': 'relu',
    'trainable': True,
    'name': 'activation_10'}},
  {'class_name': 'BatchNormalization',
   'config': {'gamma_regularizer': None,
    'name': 'batchnormalization_10',
    'epsilon': 0.001,
    'trainable': True,
    'mode': 0,
    'beta_regularizer': None,
    'momentum': 0.99,
    'axis': -1}},
  {'class_name': 'Dropout',
   'config': {'p': 0.1, 'trainable': True, 'name': 'dropout_10'}},
  {'class_name': 'Convolution1D',
   'config': {'W_constraint': None,
    'b_constraint': None,
    'name': 'convolution1d_11',
    'activity_regularizer': None,
    'trainable': True,
    'filter_length': 5,
    'init': 'glorot_uniform',
    'bias': True,
    'nb_filter': 120,
    'input_dim': None,
    'subsample_length': 1,
    'border_mode': 'valid',
    'b_regularizer': None,
    'W_regularizer': None,
    'activation': 'linear',
    'input_length': None}},
  {'class_name': 'Activation',
   'config': {'activation': 'relu',
    'trainable': True,
    'name': 'activation_11'}},
  {'class_name': 'BatchNormalization',
   'config': {'gamma_regularizer': None,
    'name': 'batchnormalization_11',
    'epsilon': 0.001,
    'trainable': True,
    'mode': 0,
    'beta_regularizer': None,
    'momentum': 0.99,
    'axis': -1}},
  {'class_name': 'Dropout',
   'config': {'p': 0.1, 'trainable': True, 'name': 'dropout_11'}},
  {'class_name': 'Convolution1D',
   'config': {'W_constraint': None,
    'b_constraint': None,
    'name': 'convolution1d_12',
    'activity_regularizer': None,
    'trainable': True,
    'filter_length': 5,
    'init': 'glorot_uniform',
    'bias': True,
    'nb_filter': 120,
    'input_dim': None,
    'subsample_length': 1,
    'border_mode': 'valid',
    'b_regularizer': None,
    'W_regularizer': None,
    'activation': 'linear',
    'input_length': None}},
  {'class_name': 'BatchNormalization',
   'config': {'gamma_regularizer': None,
    'name': 'batchnormalization_12',
    'epsilon': 0.001,
    'trainable': True,
    'mode': 0,
    'beta_regularizer': None,
    'momentum': 0.99,
    'axis': -1}},
  {'class_name': 'Activation',
   'config': {'activation': 'relu',
    'trainable': True,
    'name': 'activation_12'}},
  {'class_name': 'Dropout',
   'config': {'p': 0.1, 'trainable': True, 'name': 'dropout_12'}},
  {'class_name': 'Flatten',
   'config': {'trainable': True, 'name': 'flatten_4'}},
  {'class_name': 'Dense',
   'config': {'W_constraint': None,
    'b_constraint': None,
    'name': 'dense_4',
    'activity_regularizer': None,
    'trainable': True,
    'init': 'glorot_uniform',
    'bias': True,
    'input_dim': 15960,
    'b_regularizer': None,
    'W_regularizer': None,
    'activation': 'linear',
    'output_dim': 12}}]}

"""
