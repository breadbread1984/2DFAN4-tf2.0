#!/usr/bin/python3

from random import sample;
from math import radians,cos,sin;
from os import walk,remove;
from os.path import splitext,join,exists;
from scipy.io import loadmat;
import numpy as np;
import cv2;
import tensorflow as tf;
import pickle;

class Data(object):
    
    def __init__(self, dir = None, load_3d = True):
        
        assert dir is not None;
        assert type(load_3d) is bool;
        print('reading landmark annotaitons!');
        # load annotation to dictionary
        self.landmarks = dict();
        for root, subFolders, files in walk(dir):
            for file in files:
                fullpath = join(root, file);
                filename, ext = splitext(fullpath);
                if ext == ".mat":
                    # load pts which is a 68 x 2 matrix
                    # and transpose to 2 x 68 matrix
                    mat = loadmat(fullpath);
                    if load_3d:
                        pts = mat['pts_3d'].transpose();
                    else:
                        pts = mat['pts_2d'].transpose();
                    # derive image path
                    imagepath = filename[:-4] + ".jpg";
                    assert exists(imagepath);
                    self.landmarks[imagepath] = pts;
    
    def getAffine(self, center, s, rot, res):

        rad = radians(rot);
        # translate image to make center at the origin
        translation1 = np.eye(3);
        translation1[0,2] = -center[0];
        translation1[1,2] = -center[1];
        # couter-clockwise rotate image for rot degree
        rotate = np.eye(3);
        rotate[0,0] = cos(rad); rotate[0,1] = -sin(rad);
        rotate[1,0] = sin(rad); rotate[1,1] = cos(rad);
        # scale image
        affine = s * np.dot(rotate, translation1);
        # translate image to let the new upper left at the origin
        translation2 = np.eye(3);
        translation2[0,2] = res / 2;
        translation2[1,2] = res / 2;
        affine = np.dot(translation2, affine);
        return affine;
    
    def crop(self, img, affine, res):
        
        assert affine.shape == (3,3);
        affine_submat = affine[:2,:3];
        # crop
        return cv2.warpAffine(img,affine_submat,(res,res));
    
    def transform(self, pts, affine):
        
        assert pts.shape == (2,68);
        assert affine.shape == (3,3);
        homogeneous = np.vstack((pts,np.ones([1,pts.shape[1]],dtype = np.float32)));
        transformed = np.dot(affine,homogeneous)
        return transformed[:2,:];
    
    def drawGaussian(self, size, pt):
        
        assert len(size) == 2;
        shape = tuple(reversed(size));
        if False == (pt[0] >= 3 and pt[1] >= 3):
            return np.zeros(shape, dtype = np.float32);
        if False == (pt[0] < size[0] - 3 and pt[1] < size[1] - 3):
            return np.zeros(shape, dtype = np.float32);
        heatmap = np.zeros(shape, dtype = np.float32);
        # generate 7x7 gaussian kernel
        ksize = 7;
        kernel1D = cv2.getGaussianKernel(ksize, 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8, cv2.CV_32F);
        kernel2D = 12.054 * np.dot(kernel1D,kernel1D.transpose());
        # copy value to the subimg centered at pt
        heatmap[int(pt[1]) - 3:int(pt[1]) + 4, int(pt[0]) - 3:int(pt[0]) + 4] = kernel2D;
        return heatmap;

    def getOriginal(self, img, pts):

        center = (450 / 2,450 / 2 + 50);
        scale = 0.8;
        affine = self.getAffine(center,scale,0,256);
        # transform image
        cropped = self.crop(img, affine, 256);
        # transform key points
        transformed_pts = self.transform(pts, affine);
        return cropped, transformed_pts;

    def getAugmented(self, img, pts):

        #1) 0% scale
        s = min(max(np.random.normal() * 0.2 + 0.8,0.6),1.0);
        #2) 100% translate
        dx = min(max(-50,np.random.normal() * 25),50);
        dy = min(max(-50,np.random.normal() * 25),50);
        center = (450 / 2 + dx, 450 / 2 + 50 + dy);
        #3) 50% rotate
        r = max(min(np.random.normal() * 10,20),-20) if np.random.uniform() >= 0.5 else 0;
        affine = self.getAffine(center,s,r,256);
        # transform image
        cropped = self.crop(img, affine, 256);
        # transform key points
        transformed_pts = self.transform(pts, affine);
        #4) 20% emulate low resolution
        if np.random.uniform() <= 0.2:
            resized = cv2.resize(cropped,(96,96));
            cropped = cv2.resize(resized,(256,256));
        #5) 50% flip
        if np.random.uniform() <= 0.5:
            cropped = cv2.flip(cropped,1);
            transformed_pts[0:1,:] = cropped.shape[1] - transformed_pts[0:1,:];
        #6) 100% color augmentation
        for c in range(cropped.shape[2]):
            cropped[:,:,c] = (cropped[:,:,c] * max(min(np.random.uniform(low = 0.7, high = 1.3),1),0)).astype('uint8');
        return cropped, transformed_pts;

    def generateSample(self, img, pts, is_augmented = False):
        
        cropped, transformed_pts = self.getAugmented(img, pts) if is_augmented else self.getOriginal(img,pts);
                # generate 68 heat maps each of which corresponds to one key point
        transformed_pts /= (256 / 64);
        heatmaps = list();
        for transformed_pt in transformed_pts.transpose():
            heatmap = self.drawGaussian((64,64), transformed_pt);
            heatmaps.append(heatmap);
        heatmaps = np.array(heatmaps, dtype = np.float32);
        heatmaps = np.transpose(heatmaps, (1,2,0));
        return cropped, heatmaps;
    
    def generateTFRecord(self):
        
        print('generating trainset!');
        if True == exists('trainset.tfrecord'):
            remove('trainset.tfrecord');
        writer = tf.io.TFRecordWriter('trainset.tfrecord');
        for path, pts in self.landmarks.items():
            img = cv2.imread(path);
            if img is None: raise 'can\'t open image ' + path;
            # 1 original sample
            cropped, heatmaps = self.generateSample(img, pts);
            trainsample = tf.train.Example(features = tf.train.Features(
                feature = {
                    'data':tf.train.Feature(bytes_list = tf.train.BytesList(value = [cropped.tobytes()])),
                    'label':tf.train.Feature(float_list = tf.train.FloatList(value = heatmaps.reshape(-1)))
                }
            ));
            writer.write(trainsample.SerializeToString());
            # 9 augmented samples
            for i in range(9):
                cropped, heatmaps = self.generateSample(img, pts, True);
                trainsample = tf.train.Example(features = tf.train.Features(
                    feature = {
                        'data':tf.train.Feature(bytes_list = tf.train.BytesList(value = [cropped.tobytes()])),
                        'label':tf.train.Feature(float_list = tf.train.FloatList(value = heatmaps.reshape(-1)))
                    }
                ));
                writer.write(trainsample.SerializeToString());
        writer.close();
        
        print('generating testset!');
        if True == exists('testset.tfrecord'):
            remove('testset.tfrecord');
        writer = tf.io.TFRecordWriter('testset.tfrecord');
        for path, pts in self.landmarks.items():
            img = cv2.imread(path);
            if img is None: raise 'can\'t open image ' + path;
            # sampled augmented example
            cropped, heatmaps = self.generateSample(img, pts, True);
            trainsample = tf.train.Example(features = tf.train.Features(
                feature = {
                    'data':tf.train.Feature(bytes_list = tf.train.BytesList(value = [cropped.tobytes()])),
                    'label':tf.train.Feature(float_list = tf.train.FloatList(value = heatmaps.reshape(-1)))
                }
            ));
            writer.write(trainsample.SerializeToString());
        writer.close();
        
    def getBatch(self, batchsize = 10):
        
        landmarks = sample(self.landmarks.keys(), batchsize);
        images = list();
        labels = list();
        for landmark in landmarks:
            img = cv2.imread(landmark);
            assert img is not None;
            pts = self.landmarks[landmark];
            cropped, heatmaps = self.generateSample(img, pts, np.random.uniform() < 0.5);
            images.append(cropped);
            labels.append(heatmaps);
        images = tf.convert_to_tensor(images, dtype = tf.float32);
        labels = tf.convert_to_tensor(labels, dtype = tf.float32);
        return images, labels;

if __name__ == "__main__":
    
    data = Data('300W-LP');
    for i in range(10):
        images, labels = data.getBatch();
    data.generateTFRecord();
