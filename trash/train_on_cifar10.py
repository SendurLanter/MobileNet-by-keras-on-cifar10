import sys, os,cv2
sys.path.insert(0, os.path.abspath('..'))
from keras.callbacks import TensorBoard
import keras
import numpy as np
from model import MobileNet
batch_size = 128
num_classes = 6
epochs = 20

X_train = np.empty((2054,384,512,3),dtype="float32")
y_train=list()
X_test = np.empty((473,384,512,3),dtype="float32")
y_test=list()
def load():
	tra_i=0
	tes_i=0
	datas = os.listdir("./cardboard")
	total = len(datas)
	#print(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		#print(img.shape)
		if i <= 322:
			X_train[i] = img
			y_train.append([1])
		else:
			X_test[tes_i] = img
			y_test.append([1])
			tes_i+=1

	datas = os.listdir("./glass")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 401:
			X_train[i+322] = img
			y_train.append([2])			
		else:
			X_test[tes_i] = img
			y_test.append([2])
			tes_i+=1			

	datas = os.listdir("./metal")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 328:
			X_train[i+322+401] = img
			y_train.append([3])			
		else:
			X_test[tes_i] = img
			y_test.append([3])
			tes_i+=1			

	datas = os.listdir("./paper")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 475:
			X_train[i+322+328+401] = img
			y_train.append([4])			
		else:
			X_test[tes_i] = img
			y_test.append([4])
			tes_i+=1			

	datas = os.listdir("./plastic")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 386:
			X_train[i+322+328+401+475] =img
			y_train.append([5])			
		else:
			X_test[tes_i] = img
			y_test.append([5])
			tes_i+=1			

	datas = os.listdir("./trash")
	total = len(datas)
	for i in range(total):
		img = cv2.imread(datas[i])
		if i <= 190:
			X_train[i+322+328+401+475+386] =img
			y_train.append([0])		
		else:
			X_test[tes_i] = img
			y_test.append([0])
			tes_i+=1
	return (X_train,np.array(y_train)) , (X_test,np.array(y_test))

(x_train, y_train), (x_test, y_test) = load()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train /= 255
x_test /= 255
img_input = keras.layers.Input(shape=(384, 512, 3))
model = MobileNet(input_tensor=img_input, classes=num_classes)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1)
