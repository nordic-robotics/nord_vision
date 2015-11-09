#!/usr/bin/env python

from srv import *
from HueSatClass import HueSatClass
from cv_bridge import CvBridge, CvBridgeError
import rospy


# global hack variables
classifier = HueSatClass()
rgb_image = None
bridge = CvBridge()
LOCK = False

def handle_hue_sat(req):
	LOCK = True
    print "Returning classes"
    detected = classifier.classify( rgb_image )

    # FIND REAL RELATIVE COORDINATES ON THE PLANE

    # PROCESS DETECTED INTO A LIST OF MESSAGES
    
    classifications = []

    LOCK = False
    return HueSatResponse( classifications )

def callback(data):
	if LOCK:
		return
	try:
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
        print e

def hue_sat_server():
    rospy.init_node('hue_sat_server')
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callback, queuesize = 1)
    s = rospy.Service('hue_sat', HueSat, handle_hue_sat)
    print "Ready to classify hues and sats."
    rospy.spin()

if __name__ == "__main__":
    hue_sat_server()