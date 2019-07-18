import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model

from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten

# ** to update custom Activate functions
from keras.utils.generic_utils import get_custom_objects

from keras.datasets import cifar10
batch_size = 128
num_classes = 10
epochs = 20

""" Define layers block functions """
def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

# ** update custom Activate functions
get_custom_objects().update({'custom_activation': Activation(Hswish)})

def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE'):
    x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = BatchNormalization()(x)
    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation(Hswish)(x)
    else:
        raise NotImplementedError
    return x

def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True):
    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same')(_inputs)
    x = BatchNormalization()(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation(Hswish)(x)
    else:
        raise NotImplementedError
    return x

def __global_depthwise_block(_inputs):
    assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    kernel_size = _inputs._keras_shape[1]
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid')(_inputs)
    return x

def __se_block(_inputs, ratio=4, pooling_type='avg'):
    filters = _inputs._keras_shape[-1]
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = GlobalAveragePooling2D()(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs)
    else:
        raise NotImplementedError
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([_inputs, se])

def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_ratio, is_use_bais=False, shortcut=True, is_use_se=True, activation='RE', *args):
    # ** to high dim 
    bottleneck_dim = K.int_shape(_inputs)[-1] * expansion_ratio

    # ** pointwise conv 
    x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bais, activation=activation)

    # ** depthwise conv
    x = __depthwise_block(_inputs, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation)

    # ** pointwise conv
    x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut and strides == (1, 1):
        in_dim = K.int_shape(_inputs)[-1]
        if in_dim != out_dim:
            ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
            x = Add()([x, ins])
        else:
            x = Add()([x, _inputs])
    return x

def build_mobilenet_v3(input_size=224, num_classes=1000, model_type='large', pooling_type='avg', include_top=True):
    # ** input layer
    inputs = Input(shape=(input_size, input_size, 3))

    # ** feature extraction layers
    net = __conv2d_block(inputs, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS') 

    if model_type == 'large':
        config_list = large_config_list
    elif model_type == 'small':
        config_list = small_config_list
    else:
        raise NotImplementedError
        
    for config in config_list:
        net = __bottleneck_block(net, *config)
    
    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS')

    if pooling_type == 'avg':
        net = GlobalAveragePooling2D()(net)
    elif pooling_type == 'depthwise':
        net = __global_depthwise_block(net)
    else:
        raise NotImplementedError

    # ** shape=(None, channel) --> shape(1, 1, channel) 
    pooled_shape = (1, 1, net._keras_shape[-1])

    net = Reshape(pooled_shape)(net)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    
    if include_top:
        net = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
        net = Flatten()(net)
        net = Softmax()(net)

    model = Model(inputs=inputs, outputs=net)

    return model

""" define bottleneck structure """
# ** 
# **             
global large_config_list    
global small_config_list

large_config_list = [[16,  (3, 3), (1, 1), 16,  False, False, False, 'RE'],
                     [24,  (3, 3), (2, 2), 64,  False, False, False, 'RE'],
                     [24,  (3, 3), (1, 1), 72,  False, True,  False, 'RE'],
                     [40,  (5, 5), (2, 2), 72,  False, False, True,  'RE'],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE'],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE'],
                     [80,  (3, 3), (2, 2), 240, False, False, False, 'HS'],
                     [80,  (3, 3), (1, 1), 200, False, True,  False, 'HS'],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS'],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS'],
                     [112, (3, 3), (1, 1), 480, False, False, True,  'HS'],
                     [112, (3, 3), (1, 1), 672, False, True,  True,  'HS'],
                     [160, (5, 5), (1, 1), 672, False, False, True,  'HS'],
                     [160, (5, 5), (2, 2), 672, False, True,  True,  'HS'],
                     [160, (5, 5), (1, 1), 960, False, True,  True,  'HS']]

small_config_list = [[16,  (3, 3), (2, 2), 16,  False, False, True,  'RE'],
                     [24,  (3, 3), (2, 2), 72,  False, False, False, 'RE'],
                     [24,  (3, 3), (1, 1), 88,  False, True,  False, 'RE'],
                     [40,  (5, 5), (1, 1), 96,  False, False, True,  'HS'],
                     [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
                     [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
                     [48,  (5, 5), (1, 1), 120, False, False, True,  'HS'],
                     [48,  (5, 5), (1, 1), 144, False, True,  True,  'HS'],
                     [96,  (5, 5), (2, 2), 288, False, False, True,  'HS'],
                     [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS'],
                     [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS']]


""" build MobileNet V3 model """
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model = build_mobilenet_v3(input_size=32, num_classes=10, model_type='large', pooling_type='avg', include_top=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1)
