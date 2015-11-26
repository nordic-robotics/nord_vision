#!/usr/bin/env python

import sys
import rospy
from nord_messages.srv import *
from nord_messages.msg import *
from std_msgs.msg import String
import numpy as np

def evidence_client(id,classification, position):
    """makes a request to the service"""
    print " wait for service"
    rospy.wait_for_service('/nord/evidence_service')
    print "done waiting"
    try:
        evidence_server = rospy.ServiceProxy('/nord/evidence_service', EvidenceSrv)
        print "before calling client"
        evidence_server(id,classification, position)
        print "after calling client"
        
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
	print "evidence!!!!"
	classification = String()
	classification.data = "kaksaasd"
	position = Vector2()	
	position.x=6
	position.y=11
	id = 1
	evidence_client(id,classification, position)

