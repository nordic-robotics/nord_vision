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
import operator

ros_image=None
bridge = CvBridge()

class ImageSub:
	def __init__(self):
		global bridge
		bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

	def callback(self,data):
		global ros_image
		ros_image = data
			

def handle_request(req):
	print "in handle"
	global ros_image
	try:
		global bridge
		rgb_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
		print "Writing to file"
		#get image
	
		#stuff i get into the service
		object_id = req.id
		classification= req.classification
		position=req.position

		#Draw an example circle on the video stream
		#cv::circle(rgb_image, cv::Point("xp", "yp"), 50, CV_RGB(255,0,0));
		print object_id, classification, position.x, position.y
		print "{}_{}_({},{}).png".format(object_id, classification.data, position.x, position.y)
		#write to file
		cv2.imwrite("{}_{}_({},{}).png".format(object_id, classification.data, position.x, position.y),rgb_image.astype('uint8'))
		print "kaka"
	except Exception, e:
		print e
	

	#stuff that does out. nothing #lol
	#return EvidenceSrvResponse()

def evidence_server():
	rospy.init_node('evidence_server_node')
	ic= ImageSub()
	s = rospy.Service('/nord/evidence_service', EvidenceSrv, handle_request)
	print "Ready to start saving images on command."
	rospy.spin()

if __name__ == "__main__":
    evidence_server()
