import sys, os,cv2
from keras.callbacks import TensorBoard
import keras
import numpy as np
from model import MobileNet
from keras.preprocessing import image
batch_size = 64
num_classes = 6
epochs = 20
'''
def load():
	PATH = os.getcwd()
	x_train = []
	y_train = []
	train_path = PATH+'/cardboard/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([1])

	train_path = PATH+'/glass/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([2])

	train_path = PATH+'/metal/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([3])

	train_path = PATH+'/paper/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([4])

	train_path = PATH+'/plastic/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([5])

	train_path = PATH+'/trash/'
	train_data = os.listdir(train_path)
	for sample in train_data:
		img_path = train_path+sample
		x = image.load_img(img_path)
		x_train.append(x)
		y_train.append([0])
	print(x_train)
	x_train = np.array(x_train)
	y_train = np.array(y_train)'''

x_train = np.empty((2527,384,512,3),dtype="float32")
x_test = np.empty((473,384,512,3),dtype="float32")
y_train=list()
y_test=list()
def load():
	tra_i=0
	tes_i=0
	datas = os.listdir('./')
	total = len(datas)
	print(total)
	for e in datas:
		img = cv2.imread(e)
		#print(tra_i)
		if e[:5] == 'cardb':
			x_train[tra_i] = img
			y_train.append([1])
			tra_i+=1
		if e[:5] == 'glass':
			x_train[tra_i] = img
			y_train.append([2])
			tra_i+=1
		if e[:5] == 'metal':
			x_train[tra_i] = img
			y_train.append([3])
			tra_i+=1
		if e[:5] == 'paper':
			x_train[tra_i] = img
			y_train.append([4])
			tra_i+=1
		if e[:5] == 'plast':
			x_train[tra_i] = img
			y_train.append([5])
			tra_i+=1
		if e[:5] == 'trash':
			x_train[tra_i] = img
			y_train.append([0])
			tra_i+=1
	return (x_train,np.array(y_train)) , (x_test,np.array(y_test))

	'''
	datas = os.listdir("./cardboard")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] = img
		y_train.append([1])
		tra_i+=1

	datas = os.listdir("./glass")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] = img
		y_train.append([2])	
		tra_i+=1

	datas = os.listdir("./metal")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] = img	
		y_train.append([3])
		tra_i+=1	

	datas = os.listdir("./paper")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] = img	
		y_train.append([4])		
		tra_i+=1

	datas = os.listdir("./plastic")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] = img	
		y_train.append([5])
		tra_i+=1

	datas = os.listdir("./trash")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		X_train[tra_i] =img	
		y_train.append([0])	
		tra_i+=1'''

(x_train, y_train), (x_test, y_test) = load()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train /= 255
x_test /= 255
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
img_input = keras.layers.Input(shape=(384, 512, 3))
model = MobileNet(input_tensor=img_input, classes=num_classes)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer = keras.optimizers.rmsprop(lr=0.0004, decay=5e-4),metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1,shuffle=True)
