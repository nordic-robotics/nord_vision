#!/usr/bin/python

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot
from time import time
import numpy.random as rnd
import cPickle
from std_msgs.msg import String


class HueSatClass:
    def __init__(self):
        self.satThresh = 50

        with open('src/nord/nord_vision/data/pixel_hue_sat/rbf_svm_g0_0001_C464158.pkl', 'rb') as fid:
            self.classifier = cPickle.load(fid)

            # This should not be hardcoed like this.
            self.classAssignments = {1:"Something yellow",
                                     2:"Something red",
                                     3:"Soisoisoisoisoisoisoisoisoisoi",
                                     4:"Something orange, could also be red",
                                     5:"Something blue",
                                     6:"Something blue",
                                     7:"Green wooden cube!",
                                     8:"Something light green"}

        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()
        self.getParams()
        self.detector = cv2.SimpleBlobDetector( self.params )
        self.nrSamples = 30


    def getParams(self):
        """Initialize parameters"""
         # Change thresholds                                                   
        self.params.minThreshold = 5
        self.params.maxThreshold = 255
        self.params.thresholdStep = int(self.params.thresholdStep)

        # Filter by Area.       
        self.params.filterByArea = True
        self.params.minArea = 250
        self.params.maxArea = 4000

        # Circularity
        self.params.filterByCircularity = True
        self.params.minCircularity = 0/100.
        self.params.maxCircularity = 100/100.

        # Convexity
        self.params.filterByConvexity = True
        self.params.minConvexity = 80/100.
        self.params.maxConvexity = 100/100.

        # Inertia
        self.params.filterByInertia = False
        # Distance
        self.minDistBetweenBlobs = 10

    def classify(self,rgb_image):
        """Assigns a class to each keypoint by sampling indices from the keypoint's bounding box and assigning it a class.
        A mojority vote decides the final classification of the object."""

        rgb_image = cv2.GaussianBlur(rgb_image, (7,7), 1)
        rgb_image[400:,200:520,:] = 0

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        satIdx = np.argwhere(hsv_image[:,:,1] < self.satThresh)
        rgb_image[satIdx[:,0], satIdx[:,1], :] = 65000
        hsv_image[satIdx[:,0], satIdx[:,1], :] = 65000
        hsv_image[400:,200:520,:] = 0
        rgb_image[400:,200:520,:] = 0

        # Detect oobjects in rgb and hsv
        hsv_keypoints = self.detector.detect( hsv_inv )
        rgb_keypoints = self.detector.detect( rgb_image )
        keypoints = hsv_keypoints + rgb_keypoints
        
        predictions = []
        for i in range(0,len(keypoints)):
            point = keypoints[i]
            size = point.size*0.7
            
            minc = int(max(0,point.pt[0] - size))
            maxc = int(min(640,minc + 2*size))
            minr = int(max(0,point.pt[1] - size))
            maxr = int(min(480,minr + 2*size))
            
            im_point = rgb_image[ minr:maxr, minc:maxc, :]
            hsv_im_point = hsv_image[ minr:maxr, minc:maxc, :]
            hue = hsv_im_point[:,:,0]
            satur = hsv_im_point[:,:,1]
            
            hue = hsv_im_point[:,:,0].flatten()
            sat = hsv_im_point[:,:,1].flatten()
            
            sampleIdx = rnd.choice( len(hue), self.nrSamples )
            data = np.transpose( np.array( [hue[sampleIdx], sat[sampleIdx]] ) )
            guessed_class = self.classifier.predict(data)
            counts = np.bincount(map(int,guessed_class))
            
            predictions.append(np.argmax(counts))

        return [ [keypoints[i].pt[0], keypoints[i].pt[1] self.classAssignments[predictions[i]] ] for i in range(len(predictions)) ]



        
