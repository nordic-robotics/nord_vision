#!/usr/bin/env python
import os
from nord_messages.srv import *
from nord_messages.msg import *
from std_msgs.msg import String
from HueSatClass import HueSatClass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
from collections import Counter
import operator
import rospkg



# global hack variables
classifier = HueSatClass()
avail = np.array([[1,0,0,0,0,0,1],[0,1,0,0,0,0,0],[1,0,0,1,0,1,1],[0,0,0,0,1,0,0],[1,0,0,0,0,0,0],[0,1,1,0,0,0,0],[0,0,0,0,0,1,0]])


def readConfusionMatrix():
    """Hardcoded reading of our confusion matrix"""
    rospack = rospkg.RosPack()
    path = rospack.get_path('nord_pointcloud')
    lines = []
    with open(os.path.join(path,'data/confusion_3nn_voxelized.txt'), 'rb') as f:
        lines = f.readlines()
        
    mat = np.array([ map(int, line.split()) for line in lines ])

    mat2 = mat[0:7,:] + mat[7:14,:] + mat[14:21,:] + mat[21:28,:] + mat[28:,:]

    summa = np.sum(mat2.astype('float'),1)
    confusion = np.transpose(np.divide(np.transpose(mat2),summa))

    return confusion

confusion = readConfusionMatrix()

objects = [ "An object", 
                "Red Cube",
                "Red Hollow Cube",
                "Blue Cube",
                "Green Cube",
                "Yellow Cube",
                "Yellow Ball",
                "Red Ball",
                "Green Cylinder",
                "Blue Triangle",
                "Purple Cross",
                "Purple Star",
                "Patric"]

classes = {     1:"yellow sphere",
                2:"red ",
                3:"pruple",
                4:"orange",
                5:"blue",
                6:"blue",
                7:"green",
                8:"light green"}
colourClassIdx = {'red':0,'pruple':1,'orange':2,'green':3,'light green':4,'blue':5,'yellow':6}
idxClassColour = {0:'red',1:'pruple',2:'orange',3:'green',4:'light green',5:'blue',6:'yellow'}
shapeClassIdx = {'ball':0,'cross':1,'cube':2,'cylinder':3,'hollowcube':4,'star':5,'triangle':6}




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


def make_a_decision(shapeArray, colourArray):
    global objects
    global classes

    print "NOW I WILL MAKE DECISION BASED ON THE SHAPE: {} AND COLOUR: {}".format(shapeArray, colourArray)

    colour = idxClassColour[ np.argmax(colourArray) ]

    # Find shapes allowed by the colour and weigh them
    availShape = np.dot(avail,color)
    # Find possibilities of getting the guessed shape and aggregate them
    confusionShapes = np.multiply(confusion[:,shape>0],shape[shape>0])
    sums = np.sum(confusionShapes,1)
    # Pick the shape allowed by the availShape from colour information
    possibleOutcomes = np.multiply(sums,availShape)
    # Choose the row: actual shape, most probable 
    guess = np.argmax(possibleOutcomes)
    shape = shapeClassIdx.keys()[guess]

    if colour=="green":
        if shape=="cube":
            return "Green Cube"
        
        return "A Green Object"

    if colour=="light green":
        if shape=="cylinder":
            return "Green Cylinder"
        
        return "Light Green Object"

    if colour=="orange":
        if shape=="star":
            return "Patric"
        
        return "Orange Object"

    if colour=="blue":
        if shape=="triangle":
            return "Blue Triangle"
        if shape=="cube":
            return "Blue Cube"

        return "Blue object"

    if colour=="yellow":
        if shape=="ball":
            return "Yellow Ball"

        return "Yellow Object"

    if colour=="pruple":
        if shape=="star":
            return "Purple Star"
        if shape=="cross":
            return "Purple Cross"

        return "Purple Object"

    if colour=="red":
        if shape=="hollowcube":
            return "Red Hollow Cube"
        if shape=="cube":
            return "Red Cube"
        if shape=="ball":
            return "Red Ball"
        if shape=="cylinder":
            return "Red Cylinder"

        return "Red Object"

def handle_request(req):
    """Makes a method call to the object detector and classifier a"""
    print "Returning classes"
    global classifier
    
    object_id = req.id

    print "call landmark for features"
    ## CALL LANDMARK SERVICE FOR FEATURES
    features = get_features_from_landmarks(object_id)

    print "classify shape"
    ## CALL FLANN SERVICE FOR SHAPE
    shape = String()
    shape.data = "???"
    shape_features = [f for f in features if len(f.vfh) > 0]

    shapeArray = np.zeros(1,7)
    if len(shape_features) > 0:
        shape_votes = get_shape_class( shape_features )
        # idx = shape_votes.counts.index( max(shape_votes.counts) )
        # shape = shape_votes.names[idx]
        for i,name in enumerate(shape_votes.names):
            shapeArray[ shapeClassIdx[ name ] ] = shape_votes.counts[i]
    shapeArray = shapeArray / np.sum(shapeArray)
    
    print "classify colour"
    ## USE CLASSIFIER FOR COLOUR
    huesats = [ np.array(f.feature).reshape(2,f.splits[0]) for f in features ]
    colours = [ classifier.classify( np.transpose( hs ) ) for hs in huesats ]

    print "sum up classifications"
    ## SUM UP THE CLASSIFICATION
    colour_votes = Counter( colours )
    print colour_votes

    colourArray = np.zeros(1,7)
    for key, value in colour_votes.iteritems():
        colourArray[colourClassIdx[key]] = value
    colourArray = colourArray / np.sum(colourArray)

    # colour = max(colour_votes.iteritems(), key=operator.itemgetter(1))[0]

    decision = make_a_decision(shapeArray, colourArray)
    # decision = make_a_decision(shape.data, colour)

    response = shape   
    response.data = decision
    return ClassificationSrvResponse( response )



def classification_server():
    rospy.init_node('classification_server_node')
    s = rospy.Service('/nord/vision/classification_service', ClassificationSrv, handle_request)
    print "Ready to classify hues and sats and shape."
    rospy.spin()

if __name__ == "__main__":
    classification_server()