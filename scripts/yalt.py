#!/usr/bin/env python
"""Yet another landmark tracker."""

import rospy
from nord_messages.msg import *
from nord_messages.srv import *
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class Yalt:
	def __init__(self,args):
		self.bridge = CvBridge()
		self.same_object_threshold = 0.40**2#m**2
		self.unique_objects = dict()
		self.id_dicts = dict()
		self.all_objects = set()
		self.object_sub = rospy.Subscriber('/nord/estimation/objects', ObjectArray, self.updateObjects, queue_size=10)
		self.unique_objects_pub = rospy.Publisher("/nord/vision/igo", ObjectArray, queue_size=20)
		self.evidence_reporter = rospy.Service('/nord/vision/prompt_evidence_reporting_service', PromptEvidenceReportingSrv, self.report_evidence)
		self.viz = "viz" in args
		print self.viz
		self.vizi_pub = rospy.Publisher('/nord/map', Marker, queue_size = 10)
		self.image_vizi_pub = rospy.Publisher('/nord/images', Image, queue_size = 1)
		print self.viz
				
	def updateObjects(self, objectArray):
		"""Filters out seen objects from the message and adds the novel ones."""
#		print "update objects"
		novelObjects = [ o for o in objectArray.data if o.id not in self.all_objects ]
		self.add( novelObjects )
		objectArray = ObjectArray()
		objectArray.data = self.unique_objects.values()
		self.unique_objects_pub.publish( objectArray )

		m = Marker()
		m.id = 74
		m.type = Marker.POINTS
		m.color.a = m.color.g = 1.0
		m.color.b = m.color.r = 0.0
		m.header.frame_id = "/map"
		m.header.stamp = rospy.get_rostime()
		m.ns = "uid"
		m.action = Marker.ADD
		m.pose.orientation.w = 1.0
		m.lifetime = rospy.Duration();
		m.scale.x = m.scale.y = 0.05;

		for o in objectArray.data:
			p = Point()
			p.x = o.x
			p.y = o.y
			p.z = 0.05
			m.points.append(p)			

		self.vizi_pub.publish(m)

	
	def add(self, novelObjects):
		"""For each novel object find others with the same classification.  
		Check whether any of them are within the same_object_threshold joins them if 
		appropriate.  Thus judging whether they are the same object."""
		for obj in novelObjects:
			self.all_objects.add(obj.id)
			obj.objectId = self.classify(obj.id)
			
			# Find similarly classified objects
			similar_objects = [ o for o in self.unique_objects.values() if o.objectId == obj.objectId ]
			
			# Find the id of a corresponding seen object
			same_id = self.find_same(obj, similar_objects)

			if same_id == -1:  # no similar object was within sam_object_threshold
				print "found: "+str(obj.id) 
				self.unique_objects[obj.id] = obj
				self.id_dicts[obj.id] = [obj.id]
				
				if self.viz:
					print "publish"
					self.image_vizi_pub.publish(obj.moneyshot)
					print "published"
			else: # Update coordinates and add the number of features
				print "updated coordinates"
				self.update_coordinates(same_id, obj)
				self.unique_objects[same_id].nrObs += obj.nrObs
				self.id_dicts[same_id].append(obj.id)


	def update_coordinates(self, same_id, obj):
		"""Title! Wheighted?"""
		print "updataed coordinates of {}".format(same_id)
		newObs = obj.nrObs
		oldObs = self.unique_objects[same_id].nrObs
		print "old coords: {} {}".format(self.unique_objects[same_id].x, self.unique_objects[same_id].y)
		self.unique_objects[same_id].x = (obj.x * newObs + self.unique_objects[same_id].x * oldObs ) / (newObs + oldObs)
		self.unique_objects[same_id].y = (obj.y * newObs + self.unique_objects[same_id].y * oldObs ) / (newObs + oldObs)



	def find_same(self, obj, similar_objects):
		"""Based on the distance to similar objects judge whether they are the same.  If the closest similar object
		is within the threshold, returns the closest similar objects id.  Otherwise returns -1 """
		same_id = -1 # Return value if it is a novel object
		distance_to_similar = [ (obj.x-so.x)**2 + (obj.y-so.y)**2  for so in similar_objects ]

		# argmin can't work with empty lists
		if len(distance_to_similar) > 0:
			closest_similar = np.argmin( distance_to_similar )
			# Judge whether they are the same
			if distance_to_similar[ closest_similar ] < self.same_object_threshold:
				same_id = similar_objects[ closest_similar ].id

		return same_id


	def classify(self, objId):
		"""Call the classifier service, which handles everything with the features.
		This may be uneccessary communication"""
		rospy.wait_for_service('/nord/vision/classification_service')   ## can we move this to init?
		try:
			classification_server = rospy.ServiceProxy('/nord/vision/classification_service', ClassificationSrv) ## and this?
			classification = classification_server(objId)

			return classification.classification
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e


	def reclassify(self, uid):
		"""Re-classify the unique object"""
		rospy.wait_for_service('/nord/estimation/landmarks_service')
		try:
			pass
		except:
			pass

	def markObjectOnImage(self, image, obj):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
		except CvBridgeError, e:
			print e
		cy = image.height / 2
		cx = image.width / 2
		cv2.putText(img, obj.objectId.data, (int(cy),int(cx)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
		return self.bridge.cv2_to_imgmsg(rgb_image.astype('uint8'), "bgr8")
		

	def report_evidence(self, request):
		"""Requests Image from Landmark tracker, attaches it to an Object message to send to 
		the evidence server and sends it."""
		print "entered reporting_evidence"
		if request.id not in self.all_objects:
			print 'id: {}, has not been seen before'.format(request.id)
			return


		### TODO: Reclassify object
		

		print "wait for moneyshot_service"
		rospy.wait_for_service('/nord/estimation/moneyshot_service')

		try:
			print "start proxy"
			moneyshot_service = rospy.ServiceProxy('/nord/estimation/moneyshot_service', MoneyshotSrv) 
			
			# create a list of id's correspondng to the unique id
			ids = self.id_dicts[request.id]
			
			# request a unified image for the id
			print "request moneyshot"
			moneyshot = moneyshot_service(ids)
			
			# construct message
			print "construct message"
			o = self.unique_objects[request.id]
			print type(o)
			o.moneyshot = markObjectOnImage( moneyshot.moneyshot, o )
			if self.viz:
				print "publish"
				self.image_vizi_pub.publish(o.moneyshot)
				print "image of a ",
				print o.objectId
				print "published"

			# request to service
			print "wait for evidence service"
			rospy.wait_for_service('/nord/evidence_service')
			print "request evidence service"
			evidence_server = rospy.ServiceProxy('/nord/evidence_service', EvidenceSrv)
			responce = evidence_server( o )
			return o.objectId.data
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		return "Reporting evidence FAILED!"



def main(args):
	rospy.init_node('YALT', anonymous=True)
	
	yalt = Yalt(args)

	rate = rospy.Rate(30)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Shutting down"

if __name__ == '__main__':
	main(sys.argv)
