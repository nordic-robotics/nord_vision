#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *
from nord_messages.msg import *
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
from ras_msgs.msg import *

def evidence_client(message):
    """makes a request to the service"""
    print " wait for service"
    rospy.wait_for_service('/nord/evidence_service')
    print "done waiting"
    try:
        evidence_server = rospy.ServiceProxy('/nord/evidence_service', EvidenceSrv)
        print "before calling client"
        evidence_server(message)
        print "after calling client"
        
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
	print "evidence!!!!"
	objectet=Object()
	classification = String()
	classification.data = "Red Cube"
	objectet.moneyshot=Image()
	objectet.objectId=classification
	evidence_client(objectet)

