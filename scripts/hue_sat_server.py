#!/usr/bin/env python

from nord_messages.srv import *
from nord_messages.msg import *
from HueSatClass import HueSatClass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy


# global hack variables
classifier = HueSatClass()
rgb_image = None
bridge = CvBridge()
LOCK = False
# waitForImage = False

def callback(data):
    """Saves the image to be used as input to detection and classification"""
    global waitForImage, LOCK
    # if not waitForImage:
    #     return
    if LOCK:
        return
    try:
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
        waitForImage = False
        LOCK = True
    except CvBridgeError, e:
        print e

def handle_hue_sat(req):
    """Makes a method call to the object detector and classifier on the image returned from the camera"""
    global waitForImage, LOCK
    # waitForImage = True
    # while waitForImage:
    #     continue
    print "Returning classes"
    LOCK = True
    #detected = classifier.classify( rgb_image )

    # FIND REAL RELATIVE COORDINATES ON THE PLANE

    # PROCESS DETECTED INTO A LIST OF MESSAGES
    classifications = ClassificationArray()
    ##### DUMMY RETURN VALUE
    c = Classification()
    c.loc.x = 0.5
    c.loc.y = 0.5
    c.name.data = "green cube"
    classifications.data.append( c )
    ######

    LOCK = False
    return HueSatResponse( classifications )



def hue_sat_server():
    rospy.init_node('hue_sat_server')
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callback, queue_size = 1)
    s = rospy.Service('hue_sat', HueSat, handle_hue_sat)
    print "Ready to classify hues and sats."
    rospy.spin()

if __name__ == "__main__":
    hue_sat_server()