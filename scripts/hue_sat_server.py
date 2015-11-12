#!/usr/bin/env python

from nord_messages.srv import *
from nord_messages.msg import *
from HueSatClass import HueSatClass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np


# global hack variables
classifier = HueSatClass()

def createClassificationMsg(obj, prediction):
    """Name tells the whole story"""
    c = Classification()
    c.loc.x = obj.x
    c.loc.y = obj.y
    c.name.data = prediction
    return c

def handle_hue_sat(req):
    """Makes a method call to the object detector and classifier on the image returned from the camera"""
    print "Returning classes"
    global classifier

    objects = req.centroids.data
    nrObjects = len(objects)
    classifications = ClassificationArray()
    
    # reformat the feature list into 2d array for each object
    objectFeatures = [ np.array(obj.feature).reshape(2,obj.splits[0]) for obj in objects ]

    # classify each object
    predictions = [ classifier.classify( features ) for features in objectFeatures ]

    # Assemble classification messages
    classification.data = [ createClassificationMsg(objects[i],predictions[i]) for i in range(nrObjects) ]

    return HueSatResponse( classifications )



def hue_sat_server():
    rospy.init_node('/nord/vision/classification_server')
    s = rospy.Service('/nord/vision/classification_service', ClassificationSrv, handle_hue_sat)
    print "Ready to classify hues and sats."
    rospy.spin()

if __name__ == "__main__":
    hue_sat_server()