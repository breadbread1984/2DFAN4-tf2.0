#!/usr/bin/python3

from copy import deepcopy;
from os.path import exists, join;
import numpy as np;
import tensorflow as tf;
import cv2;
from MTCNN import Detector;
from Model import _2DFAN4;

class Landmarker(object):
    
    def __init__(self):
        
        # landmarker
        if exists(join('models','model.h5')):
            print("load from model file...");
            self.model = tf.keras.models.load_model(join('models','model.h5'));
        elif exists('checkpoints_2DFAN4'):
            print("load from check point...");
            self.model = _2DFAN4([256,256,3]);
            optimizer = tf.keras.optimizers.Adam(1e-3);
            checkpoint = tf.train.Checkpoint(model = self.model, optimizer = optimizer, optimizer_step = optimizer.iterations);
            checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
        else:
            raise 'no way to load model!';
        # face detector
        self.detector = Detector();
   
    def expandBounding(self, bounding, size):

        center = (
            (bounding[0] + bounding[2]) / 2,
            (bounding[1] + bounding[3]) / 2 + (bounding[3] - bounding[1]) / 9
        );
        expanded = (
            int(center[0] - size[0] / 2),
            int(center[1] - size[1] / 2),
            int(center[0] + size[0] / 2),
            int(center[1] + size[1] / 2)
        );
        return expanded;

    def crop(self, img, bounding, size):
        
        assert type(size) is tuple;
        A = np.array([
            [bounding[0], bounding[0], bounding[2], bounding[2]],
            [bounding[1], bounding[3], bounding[3], bounding[1]],
            [1,           1,           1,           1]
        ], dtype = np.float32);
        B = np.array([
            [0, 0,          size[0],    size[0]],
            [0, size[1],    size[1],    0],
            [1, 1,          1,          1]
        ], dtype = np.float32);
        AAt = np.dot(A,A.transpose());
        ABt = np.dot(A,B.transpose());
        u,s,v = np.linalg.svd(AAt);
        AAt_inv = np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()));
        affine = np.dot(AAt_inv,ABt).transpose();
        affine = affine[:2,:];
        patch = cv2.warpAffine(img, affine, size);
        return patch;
    
    def postprocess(self, heatmaps):
        
        pts = list();
        for heatmap in np.transpose(heatmaps,(2,0,1)):
            # argmax of heatmap
            pt = np.array(np.unravel_index(heatmap.argmax(),heatmap.shape)).astype('float32');
            # refine position
            if 0 < pt[0] and pt[0] < 63 and 0 < pt[1] and pt[1] < 63:
                if heatmap[int(pt[0]) + 1, int(pt[1])] > heatmap[int(pt[0]) - 1, int(pt[1])]:
                    pt[0] = pt[0] + 0.25;
                elif heatmap[int(pt[0]) + 1, int(pt[1])] < heatmap[int(pt[0]) - 1, int(pt[1])]:
                    pt[0] = pt[0] - 0.25;
                if heatmap[int(pt[0]), int(pt[1]) + 1] > heatmap[int(pt[0]), int(pt[1]) - 1]:
                    pt[1] = pt[1] + 0.25;
                elif heatmap[int(pt[0]), int(pt[1]) + 1] < heatmap[int(pt[0]), int(pt[1]) - 1]:
                    pt[1] = pt[1] - 0.25;
            # image coordinate is the reversed coordinate of matrix
            pt = tuple(reversed(pt * 4));
            pts.append(pt);
        # return 2 x 68 matrix
        return np.array(pts,dtype = np.float32).transpose();
    
    def project(self, landmarks, size):
        
        assert type(size) is tuple;
        center = np.reshape(np.array(size, dtype = np.float32) / 2,(2,1));
        scale = np.reshape(np.array(size, dtype = np.float32),(2,1));
        return (landmarks - center) / scale;
    
    def reproject(self, landmarks_proj, bounding):
        
        center = np.reshape(
            np.array(
                [(bounding[0] + bounding[2]) / 2, (bounding[1] + bounding[3]) / 2],
                dtype = np.float32
            ), (2,1));
        scale = np.reshape(
            np.array(
                [bounding[2] - bounding[0], bounding[3] - bounding[1]],
                dtype = np.float32
            ), (2,1));
        return landmarks_proj * scale + center;
    
    def landmark(self, rgb):

        with tf.device('/cpu:0'):
            faces = self.detector.detect(rgb);
        retval = list();
        for face in faces:
            upper_left = tuple(face[0:2]);
            down_right = tuple(face[2:4]);
            # crop a square area centered at face
            length = int(1.5 * max(down_right[0] - upper_left[0],down_right[1] - upper_left[1]));
            face = self.expandBounding(face,(length,length));
            faceimg = self.crop(rgb,face,(length, length));
            faceimg_rz = cv2.resize(faceimg,(256,256));
            faceimg_rz = faceimg_rz[np.newaxis,...].astype(np.float32);
            heatmaps = self.model.predict(faceimg_rz,batch_size = 1);
            heatmaps = heatmaps[0,...];
            landmarks = self.postprocess(heatmaps);
            landmarks_proj = self.project(landmarks, (256,256));
            landmarks_reproj = self.reproject(landmarks_proj, face);
            retval.append(landmarks_reproj);
        return retval;
    
    def visualize(self, rgb, landmarks):
 
        img = deepcopy(rgb);
        for landmark in landmarks:
            for pt in landmark.transpose():
                pt = tuple(pt.astype('int32'));
                cv2.circle(img, pt, 2, (0,255,0), -1);
        return img;

if __name__ == "__main__":
   
    import sys;
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <video>");
        exit(0);
    landmarker = Landmarker();
    cap = cv2.VideoCapture(sys.argv[1]);
    if cap is None:
        print('invalid video!');
        exit(0);
    wri = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), cap.get(cv2.CAP_PROP_FPS), \
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))));
    while True:
        ret, img = cap.read();
        if ret == False: break;
        show = landmarker.visualize(img,landmarker.landmark(img));
        cv2.imshow('show',show);
        wri.write(show);
        cv2.waitKey(int(1000/cap.get(cv2.CAP_PROP_FPS)));
    wri.close();

