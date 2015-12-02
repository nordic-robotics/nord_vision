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
from scipy.spatial.distance import cdist
import rospkg
import os
import message_filters

from nord_messages.msg import CoordinateArray, Coordinate

class ImageObjectFilter:
    def __init__(self, arg):
        self.bridge = CvBridge()
        self.viz = arg == "viz"
        self.pub = arg == "pub"
        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()
        rospack = rospkg.RosPack()
        path = rospack.get_path('nord_vision')
        
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.pcl_CoordinateArray_sub = message_filters.Subscriber("/nord/pointcloud/centroids", CoordinateArray)

        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pcl_CoordinateArray_sub], queue_size = 10, slop = 0.5)
        self.synchronizer.registerCallback(self.detectAndFilter)
        
        self.ugo_CoordinateArray_pub = rospy.Publisher("/nord/vision/ugo", CoordinateArray, queue_size=20)
        if self.pub:
            self.blobImage_pub = rospy.Publisher("/nord/vision/blobs", Image, queue_size=20)

        self.boundingBoxScale = 0.7
        self.blob_dist_scale = 2
        self.nrSamples = 30

        rospack = rospkg.RosPack()
        path = rospack.get_path('nord_vision')
        self.lowerCropValue = 450  # we should calculate this from tilt angle
        self.calibrationAngle, self.calibrationHeight, self.upperCropValue = self.readCalibration(os.path.join(path,"../nord_pointcloud/data/calibration.txt"))

        print self.upperCropValue
        print self.calibrationAngle
 
        #Create trackbars for some parametersrosro
        #cv2.namedWindow('keypoints',cv2.WINDOW_NORMAL)
        if self.viz:
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
            cv2.createTrackbar('minDistance','bars', 42, 255, self.nothing)

            cv2.createTrackbar('sat','bars', 160, 255, self.nothing)
            self.getParams()
        else:
            self.params.minThreshold = 5
            self.params.maxThreshold = 255
            self.params.thresholdStep = 10

            # Filter by Area.       
            self.params.filterByArea = True
            self.params.minArea = 500
            self.params.maxArea = 4000

            # Circularity
            self.params.filterByCircularity = True
            self.params.minCircularity = 0/100.
            self.params.maxCircularity = 100/100.

            # Convexity
            self.params.filterByConvexity = True
            self.params.minConvexity = 80/100.
            self.params.maxConvexity = 100/100.

            # Inertia
            self.params.filterByInertia = False

            self.params.minDistBetweenBlobs = 42
            self.params.filterByColor = False;


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
        self.params.filterByColor = False;
        # Distance
        self.params.minDistBetweenBlobs = cv2.getTrackbarPos('minDistance','bars')

    # def joinBlobs(self, rgb_blobs, hsv_blobs):
    #     """"""

    #     blobs = np.array( [ [ blob.pt[0], blob.pt[1], blob.size ] for blob in (rgb_blobs + hsv_blobs) ] )
    #     dists = cdist( blobs[:,:2],  blobs[:,:2] , 'euclidean')
    #     thresholds = blobs[:,:,2]

    #     closestInd = np.argmin( dists, 1 )
    #     connected = dists[ range(blobs), closestInd ] < blobs[ closestInd, 2 ]


    def detectBlobs(self,rgb_image,hsv_image):
        """Uses simple blob detector on a smoothed rgb_image, crops away the base of the robot.
        Also draws and displays detected blobs if the the module is launched with the argument viz."""
        
        if self.viz:
            sat = cv2.getTrackbarPos('sat','bars')
        else:
            sat=160
 
        v, thresh = cv2.threshold(hsv_image[:,:,1],sat,255,cv2.THRESH_TOZERO)
        hsv_image[:,:,1] = thresh

        hsv_image[400:,200:520,:] = 0
        # rgb_image[400:,200:520,:] = 0
        # update parameters
        if self.viz:
            self.getParams()

        detector = cv2.SimpleBlobDetector( self.params )

        # We need to invert the hsv for SBD
        #hsv_image[:,:,2] = 0

        # Detect oobjects in rgb and hsv
        #rgb_keypoints = detector.detect( rgb_image )
        hsv_keypoints = detector.detect( hsv_image[:,:,1] )

        blobs = hsv_keypoints#+rgb_keypoints#self.joinBlobs(rgb_keypoints, hsv_keypoints)
        if self.viz or self.pub:
            im_with_keypoints = rgb_image
            #blobs = rgb_keypoints + hsv_keypoints
            #blobs = hsv_keypoints
            # im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, 
            #                                       rgb_keypoints, 
            #                                       np.array([]), 
            #                                       (0,0,255), 
            #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, 
                                                  hsv_keypoints, 
                                                  np.array([]), 
                                                  (255,0,0), 
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if self.viz:
                cv2.imshow("keypoints", im_with_keypoints)
                cv2.waitKey(3)

        return blobs

    def drawCentroidOnImag(self,image,centroid):
         p = (centroid.xp, centroid.yp)
         p = cv2.KeyPoint(centroid.xp,centroid.yp,100,30)
         im = cv2.drawKeypoints(image, 
                                              [p], 
                                              np.array([]), 
                                              (0,255,0), 
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         return im 

    def detectAndFilter(self, image, centroidsMessage):
        """Detects blobs and compares them to pcl centroids.  Reposts all objects detected with 
        features from both pcl and image."""
        #print "ran"

        try:
            centroidsArray = centroidsMessage
            rgb_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError, e:
            print e

        if self.viz or self.pub:
            for c in centroidsMessage.data:
                rgb_image = self.drawCentroidOnImag(rgb_image,c)
                
        # only work on the image within 2 meters
        rgb_image = rgb_image[self.upperCropValue:self.lowerCropValue,:,:]        

        rgb_image = cv2.GaussianBlur(rgb_image, (7,7), 1)

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # detect blobs
        blobs = self.detectBlobs(rgb_image,hsv_image)

        if self.pub:
            im_with_keypoints = cv2.drawKeypoints(rgb_image,
                                                  blobs,
                                                  np.array([]),
                                                  (255,0,0),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.blobImage_pub.publish(self.bridge.cv2_to_imgmsg(im_with_keypoints.astype('uint8'), "bgr8"))
            
        nrCentroids = len(centroidsMessage.data)
        nrBlobs = len(blobs)
        if nrBlobs == 0:
            return

        objectArray = CoordinateArray()
        objectArray.header = centroidsArray.header
        
        # Find color objects and their features
        boundingBoxes = [ self.getBoundingBox(blob, hsv_image, self.boundingBoxScale) for blob in blobs ]
        features = [ self.extractColorFeature(box) for box in boundingBoxes ]
        relativeCoordinates = [ self.estimateRelativeCoordinates( blob ) for blob in blobs ]

        # Convert the image to be used as a snapshot of the object
        rosimage = self.bridge.cv2_to_imgmsg(im_with_keypoints.astype('uint8'), "bgr8")

        # Construct the message to be sent with the objects
        objectArray.data = [ self.createCoordinate( blobs[i], relativeCoordinates[i], features[i], rosimage ) for  i in range( nrBlobs ) ]

        # reformat the data as arrays
        centroids = np.array( [ [ c.x, c.y, c.z, c.xp, c.yp ] for c in centroidsArray.data ] )
        blobs =     np.array( [ [ blob.pt[0], blob.pt[1] + self.upperCropValue , blob.size ] for blob in blobs ] )

        # If a centroid lies within a blob we regard it as an object and filter debris away
        if nrCentroids > 0:
            dists = cdist( centroids[:,3:],  blobs[:,:2] , 'euclidean')
            closestInd = np.argmin( dists, 1 )
            connected = dists[ range(nrCentroids), closestInd ] < blobs[ closestInd, 2 ]*self.blob_dist_scale
            for i,c in enumerate(list(connected)):
                if c:
                    print str(i) + " is connected"
                    closest = closestInd[i]
                    objectArray.data[closest].features.vfh = centroidsArray.data[i].features.vfh
                    objectArray.data[closest].hull = centroidsArray.data[i].hull

        self.ugo_CoordinateArray_pub.publish( objectArray )

    def getBoundingBox(self, point, image, scale = 1.0):
        """Crops the image around point with a scaled size"""
        size = point.size * scale
        minc = int(max(0,point.pt[0] - size))
        maxc = int(min(640,minc + 2*size))
        minr = int(max(0,point.pt[1] - size))
        maxr = int(min(480,minr + 2*size))

        #cv2.imshow("d",image[ minr:maxr, minc:maxc, :])

        return image[ minr:maxr, minc:maxc, :]

    def extractColorFeature(self, image):
        """Only hue and sat for now gaussian sampled from the center of the image. This may be unsafe
        """
        # mu = [ m/2 for m in image.shape[:2] ]
        # s = [ m/2 for m in mu ]
        # S = [[s[0] ,0],[0, s[1]]]
        # idx = np.random.multivariate_normal(mu, S, self.nrSamples ).astype(int)


        hue = image[:,:,0].flatten()
        sat = image[:,:,1].flatten()
        
        sampleIdx = rnd.choice(len(hue), self.nrSamples)
        data = np.transpose( np.array( [hue[sampleIdx], sat[sampleIdx]] ) )
        # guessed_class = self.classifier.predict(data)
        # counts = np.bincount(map(int,guessed_class))
        # print data
        # print self.classAssignments[ np.argmax(counts) ]

        #print image[:,:,0]
        return data#image[idx[:,0],idx[:,1],0:2] 

    def readCalibration(self, calibrationFile):
        """Reads the calibration file and calculates the tilt angle of the camera and sets the pixel value 
        from where to crop the image."""
        b = None
        height = None
        with open(calibrationFile,'rb') as f:
            lines = f.readlines()
            b = float(lines[1]) # the second line contains the b value
            height = float(lines[3])

        tiltAngle = np.arccos(-b)

        # crop angle
        maxAngle = np.arctan( -2. / b )
        # print maxAngle
        yAngle = 0.5 * np.pi - tiltAngle - maxAngle
        # print yAngle
        yCrop = - 240 * ( (8/np.pi) * yAngle -1 ) 
        yCrop = max(0, yCrop)
        # print yCrop

        return tiltAngle, height, yCrop

    def estimateRelativeCoordinates(self, blob):
        """ Returns an estimate of relative coordinates (X,Y) of a blob from it's
         image coordinates (x,y).  X is along the face of the robot, Y to the sides."""
        x = blob.pt[0]
        y = min(479, blob.pt[1] + blob.size + self.upperCropValue) # plus because high y values mean lower in the image

        # Angles relative to the camera orientation
        xAngle = -(np.pi/6) * (x - 320) / 320
        yAngle = -(np.pi/8) * (y - 240) / 240

        # The angle, the ground/camera/object form with Y kept at zero
        theta = np.pi/2 - self.calibrationAngle + yAngle
        
        # The actual X and Y relative coordinates
        X = np.tan(theta) * self.calibrationHeight
        Y = X * np.tan(xAngle)  
        #print [X,Y]
        return [ X, Y ]

    def createCoordinate(self, blob, relativeCoordinates, feature, image):
        """Assemblers a Coordinate message"""
        c = Coordinate()
        c.x = relativeCoordinates[0]
        c.y = relativeCoordinates[1]
        c.z = 0

        c.xp = int(blob.pt[0])
        c.yp = int(blob.pt[1]) + self.upperCropValue

        c.features.feature = list(feature[:,0].flatten()) + list(feature[:,1].flatten())
        c.features.splits = [len(c.features.feature)/2] # length will never be odd
        c.moneyshot = image
        return c
                
def main(args):

    rospy.init_node('ImageObjectFilter', anonymous=True)
    
    detector = ImageObjectFilter(args[-1])

    rate = rospy.Rate(30)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"

if __name__ == '__main__':
    main(sys.argv)
