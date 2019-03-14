#!/usr/bin/python3

from copy import deepcopy;
from os.path import exists;
import tensorflow as tf;
import cv2;
import dlib;
from Model import Landmark_2DFAN4;

class Landmarker(object):
    
    def __init__(self):
        
        # landmarker
        self.model = Landmark_2DFAN4();
        if exists('checkpoints_2DFAN4'):
            optimizer = tf.keras.optimizers.Adam(1e-3);
            checkpoint = tf.train.Checkpoint(model = self.model, optimizer = optimizer, optimizer_step = optimizer.iterations);
            checkpoint.restore(tf.train.latest_checkpoint('checkpoints_2DFAN4'));
        elif exists('model'):
            self.model.load_weights('model/2dfan4');
        else:
            raise 'no way to load model!';
        # face detector
        self.detector = dlib.get_frontal_face_detector();
    
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
        return path;
    
    def postprocess(self, heatmaps):
        
        pts = list();
        for heatmap in np.transpose(heatmaps,(2,0,1)):
            # argmax of heatmap
            pt = (np.unravel_index(heatmap.argmax(),heatmap.shape)).astype('float32');
            # refine position
            if 0 < pt[0] and pt[0] < 63 and 0 < pt[1] and pt[1] < 63:
                if heatmap[pt[0] + 1, pt[1]] > heatmap[pt[0] - 1, pt[1]]:
                    pt[0] = pt[0] + 0.25;
                elif heatmap[pt[0] + 1, pt[1]] < heatmap[pt[0] - 1, pt[1]]:
                    pt[0] = pt[0] - 0.25;
                if heatmap[pt[0], pt[1] + 1] > heatmap[pt[0], pt[1] - 1]:
                    pt[1] = pt[1] + 0.25;
                elif heatmap[pt[0], pt[1] + 1] < heatmap[pt[0], pt[1] - 1]:
                    pt[1] = pt[1] - 0.25;
            # image coordinate is the reversed coordinate of matrix
            pt = tuple(reversed(pt * 4));
            pts.append(pt);
        # return 2 x 68 matrix
        return np.array(pts,dtype = np.float32).transpose();
    
    def project(self, landmarks, size):
        
        assert type(size) is tuple;
        center = np.array(size / 2, dtype = np.float32);
        scale = np.array(size, dtype = np.float32);
        return (landmarks - center) / scale;
    
    def reproject(self, landmarks_proj, bounding):
        
        center = ((bounding.left() + bounding.right()) // 2, (bounding.top() + bounding.bottom()) // 2);
        scale = np.array([bounding.right() - bounding.left(), bounding.bottom() - bounding.top()], dtype = np.float32);
        return landmarks_proj * scale + center;
    
    def landmark(self, rgb):

        faces = self.detector(rgb,0);
        retval = list();
        for face in faces:
            # crop a square area centered at face
            length = 1.2 * max(face.right() - face.left(),face.bottom() - face.top());
            faceimg = self.crop(rgb,face,(length, length));
            faceimg_rz = cv2.resize(faceimg,(256,256));
            heatmaps = self.model.predict(faceimg_rz,batch_size = 1).numpy();
            landmarks = self.postprocess(heatmaps);
            landmarks_proj = self.project(landmarks, (256,256));
            landmarks_reproj = self.reproject(landmarks_proj, face);
            retval.append(landmarks_reproj);
        return retval;
    
    def visualize(self, rgb, landmarks):
 
        img = deepcopy(rgb);
        for landmark in landmarks:
            for pt in landmark.transpose():
                cv2.circle(img, pt, 2, (0,255,0), -1);
        return img;

if __name__ == "__main__":
    
    landmark = Landmark();
    img = cv2.imread('test/christmas.jpg');
    show = landmark.visualize(img,landmark.landmark(img));
    cv2.imshow('show',show);
    cv2.waitKey();

