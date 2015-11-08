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
import scipy as sp
from scipy.spatial.distance import cdist

from nord_messages.msg import CoordinateArray, Coordinate

class ImageObjectFilter:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.storeBlobs, queue_size = 1)
        self.pcl_CoordinateArray_sub = rospy.Subscriber("/coord", CoordinateArray, self.storeCentroids, queue_size = 1)
        self.ugo_CoordinateArray_pub = rospy.Publisher("/UGO", CoordinateArray, queue_size=20)
        # self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.displayDepth, queue_size = 1)
        # cv2.namedWindow('depth',cv2.WINDOW_NORMAL)

        self.blobs = np.array([])
        self.centroidsArray = np.array([])
        self.keypoints=[]
        self.minDistToConnect = 10   # in pixels
        self.LOCKED = False
        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()
 
        #Create trackbars for some parametersrosro
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

        cv2.createTrackbar('sat','bars', 40, 255, self.nothing)

    def nothing(self, x):
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

    def storeBlobs(self, data):
        """Detects blobs and records them for comparison to the centroids"""
        
        if self.LOCKED:
            return
        print "store blobs called"
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e
            
        rgb_image = cv2.GaussianBlur(rgb_image, (7,7), 1)
        

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        sat = cv2.getTrackbarPos('sat','bars')
        satIdx = np.argwhere(hsv_image[:,:,1] < sat)
        rgb_image[satIdx[:,0], satIdx[:,1], :] = 65000
        hsv_image[satIdx[:,0], satIdx[:,1], :] = 65000
        hsv_image[400:,200:520,:] = 0
        rgb_image[400:,200:520,:] = 0
        # update parameters
        self.getParams()
        

        detector = cv2.SimpleBlobDetector( self.params )

        # We need to invert the hsv for SBD
        hsv_inv = -hsv_image

        # Detect oobjects in rgb and hsv
        rgb_keypoints = detector.detect( rgb_image )
        hsv_keypoints = detector.detect( hsv_inv )
        self.blobs = rgb_keypoints
        im_with_keypoints = cv2.drawKeypoints(rgb_image, 
                                              rgb_keypoints, 
                                              np.array([]), 
                                              (0,255,0), 
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, 
        #                                       hsv_keypoints, 
        #                                       np.array([]), 
        #                                       (255,0,0), 
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints", im_with_keypoints)
        cv2.waitKey(3)

    # def displayDepth(self,data):
    #     depth_image=0
    #     try:
    #         depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")

    #     except CvBridgeError, e:
    #         print e

    #     depth_image[depth_image < 300] = 60001
    #     depth_image = np.uint8(depth_image/(2.**8))
    #     depth_image = cv2.GaussianBlur(depth_image, (7,7), 1)
        
    #     detector = cv2.SimpleBlobDetector( self.params )
    #     self.keypoints = detector.detect( -depth_image )
        
    #     print "keypoints: {}".format(len(self.keypoints))

    #     cv2.imshow("depth", depth_image)
    #     cv2.waitKey(3)

    def storeCentroids(self, data ):
        """Records the centroids from the pointcloud"""
    
        if self.LOCKED:
            return
        print "store centroids called"
        try:
            self.centroidsArray = data
        except e:
            print e

    def detect(self):
        """Compares blobs to centroids.  Centroids that do not correspond to a blob will 
        be considered debris.  While it is doing this it locks the callback functions from doing their job."""

        nrCentroids = len(self.centroidsArray.data)
        if len(self.blobs) == 0 or nrCentroids == 0:
            return
        self.LOCKED = True

        # reformat the data as arrays
        centroids = np.array( [ [ c.x, c.y, c.z, c.xp, c.yp ] for c in self.centroidsArray.data ] )
        blobs =     np.array( [ [ blob.pt[0], blob.pt[1], blob.size ] for blob in self.blobs ] ) 
        
        # If a centroid lies within a blob we regard it as an object and filter debris away
        dists = cdist( centroids[:,3:],  blobs[:,:2] , 'euclidean')
        closestInd = np.argmin( dists, 1 )
        connected = dists[range(nrCentroids),closestInd] < blobs[closestInd,2]
        self.centroidsArray.data = [self.centroidsArray.data[ idx ] for idx in range(nrCentroids) if connected[idx]]

        # publish the filtered objects
        self.ugo_CoordinateArray_pub.publish( self.centroidsArray )

        self.centroidsArray = CoordinateArray()
        self.blobs = np.array([])
        self.LOCKED = False


                
def main(args):

    rospy.init_node('ImageObjectFilter', anonymous=True)
    
    detector = ImageObjectFilter()

    rate = rospy.Rate(30)

    try:
        while not rospy.is_shutdown():
            detector.detect()
            rate.sleep()
    except KeyboardInterrupt:
        print "Shutting down"

if __name__ == '__main__':
    main(sys.argv)
