#!/usr/bin/python3

import os.path;
import numpy as np;
import cv2;
import tensorflow as tf;
from Model import _2DFAN4;
from Data import Data;
from Landmarker import Landmarker;

batch_size = 10;

def main():
    
    model = _2DFAN4([256,256,3]);
    data = Data('300W-LP');
    optimizer = tf.keras.optimizers.Adam(1e-4);
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
    log = tf.summary.create_file_writer('checkpoints_2DFAN4');
    avg_loss = tf.keras.metrics.Mean(name = "loss", dtype = tf.float32);
    while True:
        images, labels = data.getBatch(batch_size);
        with tf.GradientTape() as tape:
            heatmaps = model(images);
            dists = tf.math.squared_difference(heatmaps,labels);
            total_dists = tf.math.reduce_sum(dists, axis = (1,2,3));
            loss = tf.math.reduce_mean(total_dists, axis = (0));
            avg_loss.update_state(loss);
        if avg_loss.result() < 0.01: break;
        if tf.equal(optimizer.iterations % 1, 0):
            with log.as_default():
                tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
            print('Step #%d Loss: %.6f' % (optimizer.iterations,avg_loss.result()));
            avg_loss.reset_states();
        grads = tape.gradient(loss, model.trainable_variables);
        optimizer.apply_gradients(zip(grads, model.trainable_variables));
        if tf.equal(optimizer.iterations % 100, 0):
            checkpoint.save(os.path.join('checkpoints_2DFAN4','ckpt'));
            landmarker = Landmarker();
            img = cv2.imread('test/christmas.jpg');
            if img is not None:
                show = landmarker.visualize(img,landmarker.landmark(img));
                with log.as_default():
                    show = np.expand_dims(show, axis = 0);
                    tf.summary.image('landmark', show, step = optimizer.iterations);
    # save final model
    model.save('model.h5');

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
