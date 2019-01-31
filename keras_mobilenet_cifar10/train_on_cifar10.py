import sys, os
sys.path.insert(0, os.path.abspath('..'))
from keras.callbacks import TensorBoard
import keras
from keras.datasets import cifar10
from mobilenet import MobileNet
batch_size = 128
num_classes = 10
epochs = 20
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
img_input = keras.layers.Input(shape=(32, 32, 3))
model = MobileNet(input_tensor=img_input, classes=num_classes)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1,
          callbacks=[TensorBoard(log_dir='./runs/keras',write_grads=True)])
