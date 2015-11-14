#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *
import numpy as np

def hue_sat_client():
    """makes a request to the service"""
    rospy.wait_for_service('/nord/vision/classification_service')
    try:
        hue_sat = rospy.ServiceProxy('/nord/vision/classification_service', ClassificationSrv)
        classifications = hue_sat()
        return classifications
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    print "Requesting classification"
    print hue_sat_client()