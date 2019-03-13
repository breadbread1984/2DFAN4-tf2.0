#!/usr/bin/python3

import os.path;
import tensorflow as tf;
from Model import Landmark_2DFAN4;
from Data import Data;

batch_size = 10;

def main():
    
    model = Landmark_2DFAN4();
    data = Data('300W-LP');
    optimizer = tf.keras.optimizers.Adam(1e-3);
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
    log = tf.summary.create_file_writer('checkpoints_2DFAN4');
    avg_loss = tf.keras.metrics.Mean(name = "loss", dtype = tf.float32);
    while True:
        images, labels = data.getBatch(batch_size);
        with tf.GradientTape() as tape:
            heatmaps = model(images);
            loss = tf.math.squared_difference(heatmaps,labels);
            loss = tf.math.reduce_sum(loss, axis = (1,2,3));
            loss = tf.math.reduce_mean(loss, axis = (0));
            avg_loss.update_state(loss);
        if avg_loss.result() < 0.01: break;
        if tf.equal(optimizer.iterations % 100, 0):
            with log.as_default():
                tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
            avg_loss.reset_states();
        grads = tape.gradient(loss, model.variables);
        optimizer.apply_gradients(zip(grads, model.variables));
        checkpoint.save(os.path.join('checkpoints_2DFAN4','ckpt'));
    # save final model
    if False == os.path.exists('model'): os.mkdir('model');
    model.save_weights('./model/2dfan4');

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
