#!/usr/bin/env python

from nord_messages.srv import *
from nord_messages.msg import *
from HueSatClass import HueSatClass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
from collections import Counter
import operator


# global hack variables
classifier = HueSatClass()

def createClassificationMsg(obj, prediction):
    """Name tells the whole story"""
    c = Classification()
    c.loc.x = obj.x
    c.loc.y = obj.y
    c.name.data = prediction
    return c

def get_features_from_landmarks(object_id):
    """Requests features for object id"""
    rospy.wait_for_service('/nord/estimation/landmarks_service')
    try:
        landmarks_server = rospy.ServiceProxy('/nord/estimation/landmarks_service', LandmarksSrv)
        features = landmarks_server(object_id)
        return features.data
    except rospy.ServiceException, e:
        print "LandmarksSrv call failed: %s"%e

def get_shape_class(vfh):
    """Requests classification on vfh features"""
    rospy.wait_for_service('/nord/pointcloud/shape_classification_service')
    try:
        shape_server = rospy.ServiceProxy('/nord/pointcloud/shape_classification_service', FlannSrv)
        shape = shape_server(vfh)
        print shape
        return shape
    except rospy.ServiceException, e:
        print "Shape classifications call failed: %s"%e


def handle_request(req):
    """Makes a method call to the object detector and classifier a"""
    print "Returning classes"
    global classifier

    object_id = req.id

    print "call landmark for features"
    ## CALL LANDMARK SERVICE FOR FEATURES
    features = get_features_from_landmarks(object_id)
    #print features
    #print len(features)
    # f = Features()
    # # green cube:
    # f.feature = [ 50,  50,  50,  50,  50,  50,  50,  50,  49,  49,  49,  49,  49,
    #     49,  50,  50,  50,  50,  50,  50,  50,  49,  49,  50,  50,  50,
    #     50,  50,  49,  49, 215, 212, 212, 215, 218, 221, 224, 227, 230,
    #    233, 238, 238, 238, 236, 230, 225, 222, 225, 227, 230, 230, 230,
    #    227, 222, 219, 217, 217, 222, 228, 230]
    # f.vfh = [0]*308
    # f.splits = [30]
    # features = [ f ]

    print "classify shape"
    ## CALL FLANN SERVICE FOR SHAPE
    shape_votes = get_shape_class(features)
    idx = shape_votes.counts.index( max(shape_votes.counts) )
    shape = shape_votes.names[idx]

    print "classify colour"
    ## USE CLASSIFIER FOR COLOUR
    huesats = [ np.array(f.feature).reshape(2,f.splits[0]) for f in features ]
    colours = [ classifier.classify( np.transpose( hs ) ) for hs in huesats ]

    print "sum up classifications"
    ## SUM UP THE CLASSIFICATION
    colour_votes = Counter( colours )
    print colour_votes
    colour = max(colour_votes.iteritems(), key=operator.itemgetter(1))[0]

    response = shape    
    response.data = colour + shape.data
    return ClassificationSrvResponse( response )

def classification_server():
    rospy.init_node('classification_server_node')
    s = rospy.Service('/nord/vision/classification_service', ClassificationSrv, handle_request)
    print "Ready to classify hues and sats and shape."
    rospy.spin()

if __name__ == "__main__":
    classification_server()