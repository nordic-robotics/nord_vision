#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *
from nord_messages.msg import *
import numpy as np

def classification_client(data):
    """makes a request to the service"""
    rospy.wait_for_service('/nord/vision/classification_service')
    try:
        classification_server = rospy.ServiceProxy('/nord/vision/classification_service', ClassificationSrv)
        print "before calling client"
        classification = classification_server(data)
        print "after calling client"
        return classification
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    print "Requesting classification"
    print sys.argv
    iddd = int(sys.argv[1])
    print "{}:".format(iddd)
    object_id = iddd
    print classification_client(object_id)
#    print "1:"
#    object_id = 1
#    print classification_client(object_id)
#    print "2:"
#    object_id = 2
#    print classification_client(object_id)
