
from keras.layers import Dense, Input,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.models import Model


SHAPE = (256, 256, 3)
# 25% i.e 64 width size will be mask from both side
MASK_PERCENT = .25
dis_input_shape = (SHAPE[0], int(SHAPE[1] * (MASK_PERCENT *2)), SHAPE[2])
dis_dropout = 0.25


def dis_build_conv(layer_input, filter_size, kernel_size=4, strides=2, activation='leakyrelu', dropout_rate=dis_dropout, normalization=True):
    conv = Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
    if activation == 'leakyrelu':
        conv = LeakyReLU(alpha=0.2)(conv)
    if dropout_rate:
        conv = Dropout(dropout_rate)(conv)
    if normalization == 'inst':
        conv = InstanceNormalization()(conv)
    return conv


def build_discriminator():
    dis_input = Input(shape=dis_input_shape)
    dis = dis_build_conv(dis_input, 32, 5,strides=2, normalization=False)
    for i in range(4):
        filter_size = 64
        dis = dis_build_conv(dis, filter_size, 5, strides=2)
    flatten = Flatten()(dis)
    fc1 = Dense(512, activation='relu')(flatten)
    dis_output = Dense(1, activation='sigmoid')(fc1)
    
    return Model(dis_input, dis_output)
