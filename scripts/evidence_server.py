#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
from collections import Counter
from nord_messages.srv import EvidenceSrv
from ras_msgs.msg import *
import operator
global listan
listan=['An Object', 'Red Cube','Blue Cube','Green Cube','Yellow Cube','Yellow Ball','Red Ball','Green Cylinder','Blue Triangle','Purple Cross', 'Purple Star', 'Patric', 'Red Hollow Cube'] 

global pub
pub = rospy.Publisher("/evidence", RAS_Evidence, queue_size=20)
def handle_request(req):
        global pub
	print "in handle"
	try:
		#stuff i get into the service
		classification= req.data.objectId.data
		image=req.data.moneyshot

	except Exception, e:
		print e
	
	evidence=RAS_Evidence()
	evidence.group_number=2
	evidence.stamp= rospy.Time.now()
	evidence.image_evidence=image
	if (classification in listan):
		evidence.object_id=classification
	else:
		print('this is a problem, not an option for objects!')
		return
        print "create publisher"
#	pub = rospy.Publisher("/evidence", RAS_Evidence, queue_size=20)
        print "publish"
        pub.publish( evidence )
        print "published"
	return 1337

def evidence_server():
	rospy.init_node('evidence_server_node')
	s = rospy.Service('/nord/evidence_service', EvidenceSrv, handle_request)
	print "Ready to start saving images on command."
	rospy.spin()

if __name__ == "__main__":
    evidence_server()
