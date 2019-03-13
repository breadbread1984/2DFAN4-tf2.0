#!/usr/bin/python3

import tensorflow as tf;

class ConvBlock(tf.keras.Model):

    def __init__(self,filters, preserveChannel = True):

        super(ConvBlock,self).__init__();
        assert filters % 4 == 0;
        self.filters = filters;
        self.preserveChannel = preserveChannel;
        self.bn1 = tf.keras.layers.BatchNormalization();
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv1 = tf.keras.layers.Conv2D(filters = self.filters // 2, kernel_size = (3,3), padding = 'same');
        self.bn2 = tf.keras.layers.BatchNormalization();
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2D(filters = self.filters // 4, kernel_size = (3,3), padding = 'same');
        self.bn3 = tf.keras.layers.BatchNormalization();
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2D(filters = self.filters // 4, kernel_size = (3,3), padding = 'same');
        self.concat = tf.keras.layers.Concatenate(axis = -1);
        if False == preserveChannel:
            self.bn_shortcut = tf.keras.layers.BatchNormalization();
            self.relu_shortcut = tf.keras.layers.LeakyReLU();
            self.conv_shortcut = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = (1,1), padding = 'same');
            self.add = tf.keras.layers.Add();

    def call(self, input):

        result = self.bn1(input);
        result = self.relu1(result);
        slice1 = self.conv1(result);
        result = self.bn2(slice1);
        result = self.relu2(result);
        slice2 = self.conv2(result);
        result = self.bn3(slice2);
        result = self.relu3(result);
        slice3 = self.conv3(result);
        result = self.concat([slice1,slice2,slice3]);

        if False == self.preserveChannel:
            shortcut = self.bn_shortcut(input);
            shortcut = self.relu_shortcut(shortcut);
            shortcut = self.conv_shortcut(shortcut);
            result = self.add([result, shortcut]);
        
        return result;

class HourGlass(tf.keras.Model):
    
    def __init__(self, depth = 4):
        
        super(HourGlass,self).__init__();
        self.depth = depth;
        self.cb_shortcut = ConvBlock(256);
        self.ap_downsample = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2));
        self.cb_downsample = ConvBlock(256);
        if depth > 1:
            self.hourglass = HourGlass(depth - 1);
        else:
            self.cb = ConvBlock(256);
        self.cb_upsample = ConvBlock(256);
        self.us_upsample = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest');
        self.add = tf.keras.layers.Add();
        
    def call(self, input):
        
        shortcut = self.cb_shortcut(input);
        result = self.ap_downsample(input);
        result = self.cb_downsample(result);
        if self.depth > 1:
            result = self.hourglass(result);
        else:
            result = self.cb(result);
        result = self.cb_upsample(result);
        result = self.us_upsample(result);
        result = self.add([result, shortcut]);
        
        return result;
    
class Module(tf.keras.Model):
    
    def __init__(self, isOutput = False):
        
        super(Module,self).__init__();
        self.isOutput = isOutput;
        self.hourglass = HourGlass(4);
        self.convblock = ConvBlock(256);
        self.conv1 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same');
        self.bn1 = tf.keras.layers.BatchNormalization();
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2D(filters = 68, kernel_size = (1,1), padding = 'same');
        if False == self.isOutput:
            self.conv3 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same');
            self.conv4 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), padding = 'same');
            self.add = tf.keras.layers.Add();

    def call(self, input):

        result = self.hourglass(input);
        result = self.convblock(result);
        result = self.conv1(result);
        result = self.bn1(result);
        logits = self.relu1(result);
        result = self.conv2(logits);
        if False == self.isOutput:
            a = self.conv3(logits);
            b = self.conv4(result);
            result = self.add([a,b]);
        return result;

class Landmark_2DFAN4(tf.keras.Model):
    
    def __init__(self, module_num = 4):
        
        super(Landmark_2DFAN4,self).__init__();
        self.module_num = module_num;
        self.conv = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), padding = 'same');
        self.bn = tf.keras.layers.BatchNormalization();
        self.relu = tf.keras.layers.LeakyReLU();
        self.convblock1 = ConvBlock(128,False);
        self.ap = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2));
        self.convblock2 = ConvBlock(128);
        self.convblock3 = ConvBlock(256,False);
        self.modules = list()
        for i in range(module_num):
            if i < module_num - 1:
                self.modules.append(Module(False));
            else:
                self.modules.append(Module(True));

    def call(self, input):
        
        result = self.conv(input);
        result = self.bn(result);
        result = self.relu(result);
        result = self.convblock1(result);
        result = self.ap(result);
        result = self.convblock2(result);
        result = self.convblock3(result);
        for i in range(self.module_num):
            result = self.modules[i](result);

        return result;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    model = Landmark_2DFAN4();
