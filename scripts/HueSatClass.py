#!/usr/bin/python

import sys
import os
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
import rospkg

class HueSatClass:
    def __init__(self):
        rospack = rospkg.RosPack()
        path = rospack.get_path('nord_vision')

        with open(os.path.join(path,'data/pixel_hue_sat/rbf_svm_g0_0001_C464158.pkl'), 'rb') as fid:
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

    def classify(self,features):
        """classifies each feature vector in features.
        A mojority vote decides the final classification of the object."""
        
        votes = self.classifier.predict(features)
        
        voteCounts = np.bincount(map(int,votes))
        
        majorityVote = np.argmax(voteCounts)
        
        voteLabel = self.classAssignments[majorityVote]
        
        return voteLabel
      


        
