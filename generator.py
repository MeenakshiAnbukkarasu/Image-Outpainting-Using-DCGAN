

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, Input, Conv2DTranspose, Dense, Flatten
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import ReLU, Dropout, Concatenate, BatchNormalization, Reshape
from keras_contrib.layers.normalization import InstanceNormalization
from keras.models import Model

gener_dropout = 0.25
SHAPE = (256, 256, 3)
MASK_PERCENT = .25
gener_input_shape = (SHAPE[0], int(SHAPE[1] * (MASK_PERCENT *2)), SHAPE[2])

def g_build_conv(layer_input, filter_size, kernel_size=4, strides=2, activation='leakyrelu', dropout_rate=gener_dropout, norm='inst', dilation=1):
    conv = AtrousConvolution2D(filter_size, kernel_size=kernel_size, strides=strides,atrous_rate=(dilation,dilation), padding='same')(layer_input)
    if activation == 'leakyrelu':
        conv = ReLU()(conv)
    if dropout_rate:
        conv = Dropout(dropout_rate)(conv)
    if norm == 'inst':
        conv = InstanceNormalization()(conv)
    return conv


def g_build_deconv(layer_input, filter_size, kernel_size=3, strides=2, activation='relu', dropout=0):
    deconv = Conv2DTranspose(filter_size, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
    if activation == 'relu':
        deconv = ReLU()(deconv)
    return deconv


def build_generator():
    gener_input = Input(shape=gener_input_shape)
    
    gener1 = g_build_conv(gener_input, 64, 5, strides=1)
    gener2 = g_build_conv(gener1, 128, 4, strides=2)
    gener3 = g_build_conv(gener2, 128, 4, strides=2)

    gener4 = g_build_conv(gener3, 256, 4, strides=1)
    gener5 = g_build_conv(gener4, 512, 3, strides=1, dilation=2)
    gener6 = g_build_conv(gener5, 512, 3, strides=1, dilation=4)
    gener7 = g_build_conv(gener6, 512, 3, strides=1, dilation=8)
    gener8 = g_build_conv(gener7, 512, 3, strides=1, dilation=16)
    
    gener9 = g_build_conv(gener8, 256, 4, strides=1)
    gener10 = g_build_deconv(gener9, 128, 4, strides=2)
    gener11 = g_build_deconv(gener10, 128, 4, strides=2)
    gener12 = g_build_conv(gener11, 64, 4, strides=1)
    
    gener_output = AtrousConvolution2D(3, kernel_size=4, strides=(1,1), activation='tanh',padding='same', atrous_rate=(1,1))(gener11)
    
    return Model(gener_input, gener_output)