#!/usr/bin/python

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot
from time import time
import numpy.random as rnd
import cPickle
from std_msgs.msg import String

class ObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.speaker_pub = rospy.Publisher("/espeak/string", String, queue_size=20)
        self.timeOfLastExecution = 0

                  #'src/nord/nord_vision/data/pixel_hue_sat'
        with open('src/nord/nord_vision/data/pixel_hue_sat/rbf_svm_g0_0001_C464158.pkl', 'rb') as fid:
            self.classifier = cPickle.load(fid)

            # This should not be hardcoed like this.
            self.classAssignments = {1:"Something yellow",
                                     2:"Something red",
                                     3:"Soisoisoisoisoisoisoisoisoisoi",
                                     4:"Something orange, could also be red",
                                     5:"Something blue",
                                     6:"Something blue",
                                     7:"Green wooden cube!",
                                     8:"Something light green"}

        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()
 
        #Create trackbars for some parameters
        cv2.namedWindow('keypoints',cv2.WINDOW_NORMAL)
        cv2.namedWindow('bars')
        cv2.createTrackbar('draw_hist','bars', 0, 1, self.nothing)
        
        # Thresholds
        cv2.createTrackbar('minThresh','bars', 5, 255, self.nothing)
        cv2.createTrackbar('maxThresh','bars', 255, 255, self.nothing)
        cv2.createTrackbar('step','bars', int(self.params.thresholdStep), 400, self.nothing)
        
        # Area
        cv2.createTrackbar('minArea','bars', 500, 4000, self.nothing)
        cv2.createTrackbar('maxArea','bars', 4000, 6000, self.nothing)
        
        # Convexity
        cv2.createTrackbar('minConvexity','bars', 80, 100, self.nothing)
        cv2.createTrackbar('maxConvexity','bars', 100, 100, self.nothing)

        # Circularity
        cv2.createTrackbar('minCircularity','bars', 0, 100, self.nothing)        
        cv2.createTrackbar('maxCircularity','bars', 100, 100, self.nothing)        

        # Inertia
        cv2.createTrackbar('minInertia','bars', 0, 100, self.nothing)        
        cv2.createTrackbar('maxInertia','bars', 1000, 10000, self.nothing)        
        
        # Distance
        cv2.createTrackbar('minDistance','bars', 10, 255, self.nothing)

        cv2.createTrackbar('sat','bars', 0, 255, self.nothing)

    def nothing(self,x):
        """An empty callback function"""
        pass

    def getParams(self):
        """Updates simple blob detector parameters with values on set on the trackbar"""
         # Change thresholds                                                   
        self.params.minThreshold = cv2.getTrackbarPos('minThresh','bars')
        self.params.maxThreshold = cv2.getTrackbarPos('maxThresh','bars')
        self.params.thresholdStep = cv2.getTrackbarPos('step','bars')

        # Filter by Area.       
        self.params.filterByArea = True
        self.params.minArea = cv2.getTrackbarPos('minArea','bars')
        self.params.maxArea = cv2.getTrackbarPos('maxArea','bars')

        # Circularity
        self.params.filterByCircularity = True
        self.params.minCircularity = cv2.getTrackbarPos('minCircularity','bars')/100.
        self.params.maxCircularity = cv2.getTrackbarPos('maxCircularity','bars')/100.

        # Convexity
        self.params.filterByConvexity = True
        self.params.minConvexity = cv2.getTrackbarPos('minConvexity','bars')/100.
        self.params.maxConvexity = cv2.getTrackbarPos('maxConvexity','bars')/100.

        # Inertia
        self.params.filterByInertia = False
        #print self.params.maxInertiaRatio
        #self.params.minInertiaRatio = cv2.getTrackbarPos('minIertia','bars')/100.
        #self.params.maxInertiaRatio = cv2.getTrackbarPos('maxIertia','bars')/100.

        # Distance
        self.minDistBetweenBlobs = cv2.getTrackbarPos('minDistance','bars')

    def classify(self,keypoints, rgb_image, hsv_image):
        """Assigns a class to each keypoint by sampling indices from the keypoint's bounding box and assigning it a class.
        A mojority vote decides the final classification of the object."""
        predictions = []
        for i in range(0,len(keypoints)):
            point = keypoints[i]
            size = point.size*0.7
            
            minc = int(max(0,point.pt[0] - size))
            maxc = int(min(640,minc + 2*size))
            minr = int(max(0,point.pt[1] - size))
            maxr = int(min(480,minr + 2*size))
            
            im_point = rgb_image[ minr:maxr, minc:maxc, :]
            hsv_im_point = hsv_image[ minr:maxr, minc:maxc, :]
            hue = hsv_im_point[:,:,0]
            satur = hsv_im_point[:,:,1]
            
            #cv2.imshow(str(i), im_point)
            
            hue = hsv_im_point[:,:,0].flatten()
            sat = hsv_im_point[:,:,1].flatten()
            
            sampleIdx = rnd.choice(len(hue), 30)
            data = np.transpose( np.array( [hue[sampleIdx], sat[sampleIdx]] ) )
            print data
            guessed_class = self.classifier.predict(data)
            counts = np.bincount(map(int,guessed_class))
            
            predictions.append(np.argmax(counts))

        return predictions
        

    def callback(self,data):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e
            
        d = time() - self.timeOfLastExecution
        
#        print d
        
        rgb_image = cv2.GaussianBlur(rgb_image, (7,7), 1)
        rgb_image[400:,200:520,:] = 0

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        self.getParams()
        detector = cv2.SimpleBlobDetector( self.params )

        # We need to invert the hsv for SBD
        hsv_inv = 255 - hsv_image
        # Detect oobjects in rgb and hsv
        hsv_keypoints = detector.detect( hsv_inv )
        rgb_keypoints = detector.detect( rgb_image )
        keypoints = hsv_keypoints + rgb_keypoints
        
        # Draw circles around detected objects.  Red or green if they were found in HSV or RGB respectively.
        im_with_keypoints = cv2.drawKeypoints(rgb_image, 
                                              hsv_keypoints, 
                                              np.array([]), 
                                              (0,0,255), 
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, 
                                              rgb_keypoints, 
                                              np.array([]), 
                                              (0,255,0), 
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("keypoints", im_with_keypoints)
        cv2.waitKey(3)    
        if d < 5:
            print "returned, "
            return
        self.timeOfLastExecution = time()
        
        # This function should be broken up, as it does more than classify, it also extracts features.
        guesses = self.classify( hsv_keypoints, rgb_image, hsv_image )

        counter = 1
        for guess in guesses:
            self.speaker_pub.publish(self.classAssignments[guess])
            print "Object {} is {}".format(counter,self.classAssignments[guess])
            counter+=1

        

            
def main(args):

    rospy.init_node('Object_Detector', anonymous=True)

    ic = ObjectDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"

if __name__ == '__main__':
    main(sys.argv)
