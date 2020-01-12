#!/usr/bin/python3

from os.path import join;
import numpy as np;
import cv2;
import tensorflow as tf;

class Detector(object):
    
    CELLSIZE = 12.0;
    
    def __init__(self, model_path = 'models'):
        
        self.pnet = tf.keras.models.load_model(join(model_path, 'pnet.h5'), compile = False);
        self.rnet = tf.keras.models.load_model(join(model_path, 'rnet.h5'), compile = False);
        self.onet = tf.keras.models.load_model(join(model_path, 'onet.h5'), compile = False);
    
    def calculateScales(self, img):
        
        h,w,c = img.shape;
        scale = 1.0;
        if min(w,h) > 500:
            scale = 500.0 / min(h,w);
            w = int(w * scale);
            h = int(h * scale);
        elif max(w,h) < 500:
            scale = 500.0 / max(h,w);
            w = int(w * scale);
            h = int(h * scale);
        scales = [];
        factor = 0.709;
        factor_count = 0;
        min1 = min(h,w);
        while min1 >= 12:
            scales.append(scale * pow(factor, factor_count));
            min1 *= factor;
            factor_count += 1;
        return scales;
    
    def rect2square(self, rectangles):
        
        wh = rectangles[..., 2:4] - rectangles[..., 0:2];
        l = tf.math.reduce_max(wh, axis = -1); # l.shape = (num over thres,)
        center = rectangles[..., 0:2] + wh * 0.5;
        upperleft = center - tf.stack([l,l], axis = -1) * 0.5;
        downright = upperleft + tf.stack([l,l], axis = -1);
        rectangles = tf.concat([upperleft, downright, rectangles[..., 4:]], axis = -1);
        return rectangles;
    
    def NMS(self, rectangles, threshold, type):
        
        if rectangles.shape[0] == 0: return rectangles;
        wh = rectangles[...,2:4] - rectangles[...,0:2] + tf.constant([1,1], dtype = tf.float32);
        area = wh[...,0] * wh[...,1];
        conf = rectangles[..., 4];
        I = tf.argsort(conf, direction = 'DESCENDING');
        i = 0;
        while i < I.shape[0]:
            idx = I[i];
            cur_upper_left = rectangles[idx, 0:2];
            cur_down_right = rectangles[idx, 2:4];
            cur_area = area[idx];
            following_idx = I[i+1:];
            following_upper_left = tf.gather(rectangles[...,0:2], following_idx);
            following_down_right = tf.gather(rectangles[...,2:4], following_idx);
            following_area = tf.gather(area, following_idx);
            max_upper_left = tf.math.maximum(cur_upper_left, following_upper_left);
            min_down_right = tf.math.minimum(cur_down_right, following_down_right);
            intersect_wh = min_down_right - max_upper_left;
            intersect_wh = tf.where(tf.math.greater(intersect_wh, 0), intersect_wh, tf.zeros_like(intersect_wh));
            intersect_area = intersect_wh[...,0] * intersect_wh[...,1];
            if type == 'iom':
                overlap = intersect_area / tf.math.minimum(cur_area, following_area);
            else:
                overlap = intersect_area / (cur_area + following_area - intersect_area);
            indices = tf.where(tf.less(overlap, threshold));
            following_idx = tf.gather_nd(following_idx, indices);
            I = tf.concat([I[:i+1], following_idx], axis = 0);
            i += 1;
        # result_rectangles.shape = (picked target num, 5)
        result_rectangles = tf.gather(rectangles, I);
        return result_rectangles;
    
    def detect_face_12net(self, cls_prob, roi, out_side, scale, width, height, threshold):
        
        in_side = 2 * out_side + 11;
        stride = 0;
        if out_side != 1:
            stride = float(in_side - self.CELLSIZE) / (out_side - 1);
        mask = tf.where(tf.math.greater(cls_prob, threshold), tf.ones_like(cls_prob), tf.zeros_like(cls_prob)); # mask.shape = (h,w)
        mask = tf.cast(mask, dtype = tf.bool);
        pos = tf.where(tf.math.greater(cls_prob, threshold)); # bounding.shape = (num over thres, 2)
        pos = tf.cast(tf.reverse(pos, axis = [1]), dtype = tf.float32); # in (x,y) order
        # boundingbox.shape = (num over thres, 4)
        boundingbox = tf.math.round((stride * tf.tile(pos, (1,2)) + tf.constant([0,0,11,11], dtype = tf.float32)) * scale);
        # offset.shape = (num over thres, 4)
        offset = tf.boolean_mask(roi, mask);
        score = tf.expand_dims(tf.boolean_mask(cls_prob, mask), axis = -1); # score.shape = (num over thres,1)
        boundingbox = boundingbox + offset * self.CELLSIZE * scale;
        rectangles = tf.concat([boundingbox, score], axis = 1);
        rectangles = self.rect2square(rectangles);
        rectangles = tf.concat(
            [
                tf.clip_by_value(rectangles[...,0:4], [0,0,0,0], [width - 1, height - 1, width - 1, height - 1]),
                rectangles[...,4:]
            ],
            axis = -1
        );
        wh = rectangles[...,2:4] - rectangles[...,0:2];
        # mask.shape = (num over thres, 2)
        mask = tf.where(tf.greater(wh,0), tf.ones_like(wh), tf.zeros_like(wh));
        mask = tf.cast(mask, dtype = tf.bool);
        mask = tf.math.logical_and(mask[...,0], mask[...,1]);
        pick = tf.boolean_mask(rectangles, mask);
        return self.NMS(pick, 0.3, 'iou');

    def filter_face_24net(self, cls_prob, roi, rectangles, width, height, threshold):
    
        mask = tf.where(tf.math.greater(cls_prob, threshold), tf.ones_like(cls_prob), tf.zeros_like(cls_prob)); # mask.shape = (target num,)
        mask = tf.cast(mask, dtype = tf.bool);
        boundingbox = tf.boolean_mask(rectangles[..., 0:4], mask); # boundingbox.shape = (target num, 4)
        wh = boundingbox[...,2:4] - boundingbox[...,0:2]; # wh.shape = (target num, 2)
        offset = tf.boolean_mask(roi, mask); # offset.shape = (target num, 4)
        score = tf.expand_dims(tf.boolean_mask(cls_prob, mask), axis = -1); # score.shape = (target num, 1);
        boundingbox = boundingbox + offset * tf.tile(wh, (1,2));
        rectangles = tf.concat([boundingbox, score], axis = -1);
        rectangles = self.rect2square(rectangles);
        rectangles = tf.concat(
            [
                tf.clip_by_value(rectangles[...,0:4], [0,0,0,0], [width - 1, height - 1, width - 1, height - 1]),
                rectangles[...,4:]
            ],
            axis = -1
        );
        wh = rectangles[...,2:4] - rectangles[...,0:2];
        mask = tf.where(tf.greater(wh,0), tf.ones_like(wh), tf.zeros_like(wh));
        mask = tf.cast(mask, dtype = tf.bool);
        mask = tf.math.logical_and(mask[...,0], mask[...,1]);
        pick = tf.boolean_mask(rectangles, mask);
        return self.NMS(pick, 0.3, 'iou');
    
    def filter_face_48net(self, cls_prob, roi, pts, rectangles, width, height, threshold):

        mask = tf.where(tf.math.greater(cls_prob, threshold), tf.ones_like(cls_prob), tf.zeros_like(cls_prob)); # mask.shape = (target num,)
        mask = tf.cast(mask, dtype = tf.bool);
        boundingbox = tf.boolean_mask(rectangles[..., 0:4], mask); # boundingbox.shape = (target num, 4)
        wh = boundingbox[...,2:4] - boundingbox[...,0:2]; # wh.shape = (target num, 2)
        offset = tf.boolean_mask(roi, mask); # offset.shape = (target num, 4)
        landmarks = tf.boolean_mask(pts, mask); # landmarks.shape = (target num, 10)
        score = tf.expand_dims(tf.boolean_mask(cls_prob, mask), axis = -1); # score.shape = (target num, 1)
        landmarks = landmarks * tf.concat([tf.tile(wh[...,0:1], (1,5)), tf.tile(wh[...,1:2], (1,5))], axis = -1) + \
                    tf.concat([tf.tile(boundingbox[...,0:1], (1,5)), tf.tile(boundingbox[...,1:2], (1,5))], axis = -1);
        boundingbox = boundingbox + offset * tf.tile(wh, (1,2));
        rectangles = tf.concat([boundingbox, score, landmarks], axis = -1);
        rectangles = tf.concat(
            [
                tf.clip_by_value(rectangles[...,0:4], [0,0,0,0], [width - 1, height - 1, width - 1, height - 1]),
                rectangles[...,4:]
            ],
            axis = -1
        );
        wh = rectangles[...,2:4] - rectangles[...,0:2];
        mask = tf.where(tf.greater(wh,0), tf.ones_like(wh), tf.zeros_like(wh));
        mask = tf.cast(mask, dtype = tf.bool);
        mask = tf.math.logical_and(mask[...,0], mask[...,1]);
        pick = tf.boolean_mask(rectangles, mask);
        return self.NMS(pick, 0.3, 'iom');
        
    def detect(self, img, threshold = [0.6, 0.6, 0.7]):
        
        data = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - 127.5) / 127.5;
        h,w,c = data.shape;
        inputs = tf.expand_dims(data, axis = 0);
        scales = self.calculateScales(data);
        # 1) create image pyramid and detect candidate target on image pyramid
        outs = [];
        for scale in scales:
            hs = int(h * scale);
            ws = int(w * scale);
            resized_img = tf.image.resize(inputs, (hs, ws));
            outs.append(self.pnet(resized_img));
        rectangles = tf.zeros((0,5), dtype = tf.float32);
        for i in range(len(scales)):
            # process image on i th level of image pyramid
            cls_prob = outs[i][0][0,...,1]; # cls_prob.shape = (h,w)
            roi = outs[i][1][0]; # roi.shape = (h, w, 4)
            out_h, out_w = cls_prob.shape;
            out_side = max(out_h, out_w);
            rectangle = self.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], w, h, threshold[0]);
            rectangles = tf.concat([rectangles, rectangle], axis = 0);
        rectangles = self.NMS(rectangles, 0.7, 'iou');
        
        if rectangles.shape[0] == 0: return rectangles;
    
        # 2) crop and refine
        boxes = tf.stack([rectangles[...,1], rectangles[...,0], rectangles[...,3], rectangles[...,2]], axis = -1) / tf.constant([h,w,h,w], dtype = tf.float32);
        predict_24_batch = tf.image.crop_and_resize(inputs, boxes, tf.zeros((rectangles.shape[0],), dtype = tf.int32),(24,24));
        outs = self.rnet(predict_24_batch);
        cls_prob = outs[0][...,1]; # cls_prob = (target num,)
        offset = outs[1]; # offset.shape = (target num, 4)
        rectangles = self.filter_face_24net(cls_prob, offset, rectangles, w, h, threshold[1]);

        if rectangles.shape[0] == 0: return rectangles;

        # 3) crop and refine and output landmark
        boxes = tf.stack([rectangles[...,1], rectangles[...,0], rectangles[...,3], rectangles[...,2]], axis = -1) / tf.constant([h,w,h,w], dtype = tf.float32);
        predict_batch = tf.image.crop_and_resize(inputs, boxes, tf.zeros((rectangles.shape[0],), dtype = tf.int32), (48,48));
        outs = self.onet(predict_batch);
        cls_prob = outs[0][...,1];
        offset = outs[1];
        pts = outs[2];
        rectangles = self.filter_face_48net(cls_prob, offset, pts, rectangles, w, h, threshold[2]);

        return rectangles;
    
if __name__ == "__main__":
    
    assert True == tf.executing_eagerly();
    import sys;
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' <img>');
        exit(0);
    detector = Detector();
    img = cv2.imread(sys.argv[1]);
    if img is None:
        print('invalid image!');
        exit(0);
    rectangles = detector.detect(img);
    for rectangle in rectangles:
        upper_left = tuple(rectangle[0:2]);
        down_right = tuple(rectangle[2:4]);
        conf = rectangle[4];
        landmarks = tf.stack([rectangle[5:10], rectangle[10:15]], axis = -1);
        landmarks = landmarks.numpy().astype('int32');
        cv2.rectangle(img, upper_left, down_right, (0,0,255), 3);
        for landmark in landmarks:
            cv2.circle(img, tuple(landmark), 2, (255,0,0), 2);
    cv2.imshow('detection', img);
    cv2.waitKey();
