#!/usr/bin/python

import os
import rospy
import numpy as np
from std_msgs.msg import String
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
            self.classAssignments = {1:"Yellow ",
                                     2:"Red ",
                                     3:"Pruple ",
                                     4:"Orange or red ",
                                     5:"Blue ",
                                     6:"Blue ",
                                     7:"Green ",
                                     8:"Light green "}

    def classify(self,features):
        """classifies each feature vector in features.
        A mojority vote decides the final classification of the object."""
        
        votes = self.classifier.predict(features)
        
        voteCounts = np.bincount(map(int,votes))
        
        majorityVote = np.argmax(voteCounts)
        
        voteLabel = self.classAssignments[majorityVote]
        
        return voteLabel