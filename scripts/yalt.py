#!/usr/bin/env python
"""Yet another landmark tracker."""

import rospy
from nord_messages.msg import *
import numpy as np

class Yalt:
	def __init__(self,args):
		self.same_object_threshold = 0.30**2#m**2
		self.unique_objects = dict()
		self.all_objects = set()
		self.object_sub = rospy.Subscriber('/nord/estimation/objects', ObjectArray, self.updateObjects, queue_size=10)
		self.unique_objects = rospy.Publisher("/nord/vision/igo", ObjectArray, queue_size=20)
		self.evidence_reporter = rospy.Service('/nord/estimation/report_evidence_service', ClassificationSrv, report_evidence)

	def updateObjects(self, objectArray):
		"""Filters out seen objects from the message and adds the novel ones."""
		novelObjects = [ o for o in objectArray.data if o.id not in self.all_objects ]
		self.add( novelObjects )
		objectArray = ObjectArray()
		objectArray.data = self.unique_objects.values()
		self.unique_objects.pub( objectArray )

	
	def add(self, novelObjects):
		"""For each novel object find others with the same classification.  
		Check whether any of them are within the same_object_threshold joins them if 
		appropriate.  Thus judging whether they are the same object."""
		for obj in novelObjects:
			self.all_objects.add(obj.id)
			obj.objectId = self.classify(obj)
			
			# Find similarly classified objects
			similar_objects = [ o for o in self.objects if o.objectId == obj.objectId ]
			
			# Find the id of a corresponding seen object
			same_id = self.find_same(obj, similar_objects)

			if same_id == -1:  # no similar object was within sam_object_threshold
				self.unique_objects[obj.id] = obj
			else: # Update coordinates and add the number of features
				self.update_coordinates(same_id, obj)
				self.unique_objects[same_id].nrObs += obj.nrObs
				# update coordinates

	def update_coordinates(self, same_id, obj):
		"""Title! Wheighted?"""
		newObs = obj.nrObs
		oldObs = self.unique_objects[same_id].nrObs
		obj.x = (obj.x * newObs + self.unique_objects[same_id].x * oldObs ) / (newObs + oldObs)
		obj.y = (obj.y * newObs + self.unique_objects[same_id].y * oldObs ) / (newObs + oldObs)



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


	def classify(self, obj):
		"""Call the classifier service, which handles everything with the features.
		This may be uneccessary communication"""
		rospy.wait_for_service('/nord/vision/classification_service')   ## can we move this to init?
		try:
			classification_server = rospy.ServiceProxy('/nord/vision/classification_service', ClassificationSrv) ## and this?
			classification = classification_server(data)
			return classification
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

	def report_evidence(self, request):
		"""Requests Image from Landmark tracker ... """

		rospy.wait_for_service('/nord/estimation/moneyshot_service')
		try:
			moneyshot_service = rospy.ServiceProxy('nord/estimation/moneyshot_service', REQUEST_FOR_UID) 
			# todo: create a list of id's
			# send to landmark tracking
			moneyshot = moneyshot_service(request)
			# get Image
			# construct message




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