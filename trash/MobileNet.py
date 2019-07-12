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
import cv2
X_train = np.empty((2054,384,512,3),dtype="float32")
X_test = np.empty((473,384,512,3),dtype="float32")
y_train=list()
y_test=list()
def load():
	tra_i=0
	tes_i=0
	datas = os.listdir("./cardboard")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 322:
			X_train[tra_i] = img
			y_train.append([1])
			tra_i+=1
		else:
			X_test[tes_i] = img
			y_test.append([1])
			tes_i+=1

	datas = os.listdir("./glass")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 401:
			X_train[tra_i] = img
			y_train.append([2])	
			tra_i+=1		
		else:
			X_test[tes_i] = img
			tes_i+=1	
			y_test.append([2])		

	datas = os.listdir("./metal")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 328:
			X_train[tra_i] = img	
			y_train.append([3])
			tra_i+=1
		else:
			X_test[tes_i] = img
			y_test.append([3])
			tes_i+=1			

	datas = os.listdir("./paper")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 475:
			X_train[tra_i] = img	
			y_train.append([4])		
			tra_i+=1
		else:
			X_test[tes_i] = img
			y_test.append([4])
			tes_i+=1			

	datas = os.listdir("./plastic")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 386:
			X_train[tra_i] =img	
			y_train.append([5])
			tra_i+=1		
		else:
			X_test[tes_i] = img
			y_test.append([5])
			tes_i+=1			

	datas = os.listdir("./trash")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 190:
			X_train[tra_i] =img	
			y_train.append([0])	
			tra_i+=1
		else:
			X_test[tes_i] = img
			y_test.append([0])
			tes_i+=1
	return (X_train,np.array(y_train)) , (X_test,np.array(y_test))

IMG_CHANNELS = 3
IMG_ROWS = 384
IMG_COLS = 512
BATCH_SIZE = 4
NB_EPOCH = 20
NB_CLASSES = 6
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
(X_train, y_train), (X_test, y_test) = load()
print(y_train.shape)
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES) 
#X_train = X_train.astype('float32')
x_train = X_train
x_test = X_test
X_train /= 255.
X_test /= 255.
a=1
print('sucs')
model = Sequential()
model.add(SeparableConv2D(np.int(32*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(np.int(64*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(np.int(128*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(np.int(256*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(np.int(512*a), (3, 3),init='glorot_normal' ,padding = 'same',input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr=0.0004, decay=5e-4),metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT, verbose = 1,callbacks=[TensorBoard(log_dir='./runs/keras',write_grads=True, histogram_freq=1)])
