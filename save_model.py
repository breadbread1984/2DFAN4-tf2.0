#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from Model import Landmark_2DFAN4;

if __name__ == "__main__":
    
    model = Landmark_2DFAN4();
    optimizer = tf.keras.optimizers.Adam(1e-4);
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
    model.save_weights('2DFAN4.h5', save_format = 'h5');
