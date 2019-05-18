#!/usr/bin/python3

import tensorflow as tf;

def ConvBlock(input_shape, filters, preserveChannel = True):
    
    assert filters % 4 == 0;
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.BatchNormalization()(inputs);
    results = tf.keras.layers.LeakyReLU()(results);
    slice1 = tf.keras.layers.Conv2D(filters = filters // 2, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(slice1);
    results = tf.keras.layers.LeakyReLU()(results);
    slice2 = tf.keras.layers.Conv2D(filters = filters // 4, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(slice2);
    results = tf.keras.layers.LeakyReLU()(results);
    slice3 = tf.keras.layers.Conv2D(filters = filters // 4, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.Concatenate(axis = -1)([slice1,slice2,slice3]);
    if False == preserveChannel:
        shortcut = tf.keras.layers.BatchNormalization()(inputs);
        shortcut = tf.keras.layers.LeakyReLU()(shortcut);
        shortcut = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1), padding = 'same')(shortcut);
        results = tf.keras.layers.Add()([results, shortcut]);
    return tf.keras.Model(inputs = inputs, outputs = results);

def HourGlass(input_shape, depth = 4):

    inputs = tf.keras.Input(shape = input_shape);
    shortcut = ConvBlock(inputs.shape[1:], filters = 256)(inputs);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(inputs);
    results = ConvBlock(results.shape[1:], filters = 256)(results);
    if depth > 1:
        results = HourGlass(results.shape[1:], depth = depth - 1)(results);
    else:
        results = ConvBlock(results.shape[1:], filters = 256)(results);
    results = ConvBlock(results.shape[1:], filters = 256)(results);
    results = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest')(results);
    results = tf.keras.layers.Add()([results, shortcut]);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Module(input_shape, isOutput = False):

    inputs = tf.keras.Input(shape = input_shape);
    results = HourGlass(inputs.shape[1:], depth = 4)(inputs);
    results = ConvBlock(results.shape[1:], filters = 256)(results);
    results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    logits = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 68, kernel_size = (1,1), padding = 'same')(logits);
    if False == isOutput:
        a = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same')(logits);
        b = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same')(results);
        results = tf.keras.layers.Add()([a,b]);
    return tf.keras.Model(inputs = inputs, outputs = results);
    
def _2DFAN4(input_shape, module_num = 4):

    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), padding = 'same')(inputs);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = ConvBlock(results.shape[1:], 128, False)(results);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    results = ConvBlock(results.shape[1:], 128)(results);
    results = ConvBlock(results.shape[1:], 256, False)(results);
    for i in range(module_num):
        if i < module_num - 1:
            results = Module(results.shape[1:], False)(results);
        else:
            results = Module(results.shape[1:], True)(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    model = _2DFAN4([256,256,3]);
