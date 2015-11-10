#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *

def hue_sat_client():
    """makes a request to the service"""
    rospy.wait_for_service('hue_sat')
    try:
        hue_sat = rospy.ServiceProxy('hue_sat', HueSat)
        classifications = hue_sat()
        return classifications
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    print "Requesting classification"
    print hue_sat_client()