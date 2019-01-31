from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,GlobalAveragePooling2D , AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D , SeparableConv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard
import os
import keras
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES) 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
a=1
model = Sequential()
model.add(SeparableConv2D(np.int(32*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(SeparableConv2D(np.int(64*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(SeparableConv2D(np.int(128*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(SeparableConv2D(np.int(256*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr=0.0004, decay=1e-6),metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT, verbose = 1,callbacks=[TensorBoard(log_dir='./runs/keras',write_grads=True, histogram_freq=1)])
