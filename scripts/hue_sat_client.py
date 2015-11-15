#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *
from nord_messages.msg import *
import numpy as np

def hue_sat_client(data):
    """makes a request to the service"""
    rospy.wait_for_service('/nord/vision/classification_service')
    try:
        hue_sat = rospy.ServiceProxy('/nord/vision/classification_service', ClassificationSrv)
        print "before calling client"
        classifications = hue_sat(data)
        print "after calling client"
        return classifications
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    print "Requesting classification"
    coords = CoordinateArray()
    coord = Coordinate()
    coord.feature = [1600 for i in range(30)] + [0 for j in range(30)]
    coord.splits = [30]
    coords.data.append(coord)

    print hue_sat_client(coords)