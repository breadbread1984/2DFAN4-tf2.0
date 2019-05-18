#!/usr/bin/python3

from copy import deepcopy;
from os.path import exists;
import numpy as np;
import tensorflow as tf;
import cv2;
import dlib;
from Model import _2DFAN4;

class Landmarker(object):
    
    def __init__(self):
        
        # landmarker
        if exists('model.h5'):
            print("load model from weight directory");
            self.model = tf.keras.models.load_model('model.h5');
        elif exists('checkpoints_2DFAN4'):
            print("load model from check point...");
            self.model = _2DFAN4([256,256]);
            optimizer = tf.keras.optimizers.Adam(1e-3);
            checkpoint = tf.train.Checkpoint(model = self.model, optimizer = optimizer, optimizer_step = optimizer.iterations);
            checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
        else:
            raise 'no way to load model!';
        # face detector
        self.detector = dlib.get_frontal_face_detector();
   
    def expandBounding(self, bounding, size):

        center = (
            (bounding.left() + bounding.right()) / 2,
            (bounding.top() + bounding.bottom()) / 2 + (bounding.bottom() - bounding.top()) / 9
        );
        expanded = dlib.rectangle(
            int(center[0] - size[0] / 2),
            int(center[1] - size[1] / 2),
            int(center[0] + size[0] / 2),
            int(center[1] + size[1] / 2)
        );
        return expanded;

    def crop(self, img, bounding, size):
        
        assert type(size) is tuple;
        A = np.array([
            [bounding.left(),   bounding.left(),    bounding.right(),   bounding.right()],
            [bounding.top(),    bounding.bottom(),  bounding.bottom(),  bounding.top()],
            [1,                 1,                  1,                  1]
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
                [(bounding.left() + bounding.right()) / 2, (bounding.top() + bounding.bottom()) / 2],
                dtype = np.float32
            ), (2,1));
        scale = np.reshape(
            np.array(
                [bounding.right() - bounding.left(), bounding.bottom() - bounding.top()],
                dtype = np.float32
            ), (2,1));
        return landmarks_proj * scale + center;
    
    def landmark(self, rgb):

        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);
        faces = self.detector(gray,0);
        retval = list();
        for face in faces:
            # crop a square area centered at face
            length = int(1.5 * max(face.right() - face.left(),face.bottom() - face.top()));
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
    
    landmarker = Landmarker();
    img = cv2.imread('test/christmas.jpg');
    assert img is not None;
    show = landmarker.visualize(img,landmarker.landmark(img));
    cv2.imshow('show',show);
    cv2.waitKey();

