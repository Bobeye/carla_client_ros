#!/usr/bin/env python


import rospy
import numpy as np
from std_msgs.msg import Float64, String, Header
from geometry_msgs.msg import TwistWithCovarianceStamped, PolygonStamped, Point32
from geometry_msgs.msg import Polygon as rosPolygon
from geometry_msgs.msg import Point as rosPoint
from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from shapely.geometry import Polygon, Point
import json
import struct
import time
import math
import cv2
import tf
import glob
import csv

from carla_ros_msgs.msg import TrafficLight, Pedestrian, Vehicle, TrafficLights, Pedestrians, Vehicles

class pcam():

	def __init__(self, csvpath=None):
		self.csvpath = csvpath
		self.init_reachability()

		self.frequency = 10
		self.current_velocity = None
		self.current_heading = None
		self.nearest_pedestrian = None

		rospy.init_node('reachability_node', anonymous=True)

		# launch subscriber
		rospy.Subscriber('/carla/ego_speed', Float64, self.ego_speed_cb)
		rospy.Subscriber('/carla/pedestrians', Pedestrians, self.pedestrians_cb)
		rospy.Subscriber('/carla/ego_pose', Pose, self.ego_pose_cb)

		# launch publisher
		self.critical_polygon_pub = rospy.Publisher("/reachability/critical", PolygonStamped, queue_size=1)
		self.imminent_polygon_pub = rospy.Publisher("/reachability/imminent", PolygonStamped, queue_size=1)

		self.main_loop()

	def main_loop(self):
		rate = rospy.Rate(self.frequency)
		while not rospy.is_shutdown():
			
			if self.current_velocity is not None and self.current_heading is not None and self.nearest_pedestrian is not None:
				x = self.nearest_pedestrian[0]/100
				y = self.nearest_pedestrian[1]/100
				v = self.current_velocity
				if v > 20:
					v = 20

				h = self.current_heading
				state = [x,y,v,h]
				vi, ci, ii = self.check_reachability(state)
				
				critical_polygon = PolygonStamped()
				header = Header()
				header.stamp = rospy.Time.now()
				header.frame_id = 'velodyne'
				critical_polygon.header = header
				for i in range(len(self.critical_reachability[vi][ci][0])):
					x = self.critical_reachability[vi][ci][0][i]
					y = self.critical_reachability[vi][ci][1][i]
					xr = x*np.cos(h)-y*np.sin(h)
					yr = x*np.sin(h)+y*np.cos(h)
					z = 0
					critical_polygon.polygon.points.append(Point32(x=xr, y=yr, z=z))
				self.critical_polygon_pub.publish(critical_polygon)

				imminent_polygon = PolygonStamped()
				header = Header()
				header.stamp = rospy.Time.now()
				header.frame_id = 'velodyne'
				imminent_polygon.header = header
				for i in range(len(self.imminent_reachability[vi][ii][0])):
					x = self.imminent_reachability[vi][ii][0][i]
					y = self.imminent_reachability[vi][ii][1][i]
					xr = x*np.cos(h)-y*np.sin(h)
					yr = x*np.sin(h)+y*np.cos(h)
					z = 0
					imminent_polygon.polygon.points.append(Point32(x=xr, y=yr, z=z))
				self.imminent_polygon_pub.publish(imminent_polygon)

			
				
			rate.sleep()

	def ego_speed_cb(self, data):
		self.current_velocity = data.data

	def pedestrians_cb(self, data):
		ped_positions = []
		for ped in data.pedestrians:
			ped_positions += [[ped.position.x, ped.position.y, np.sqrt(ped.position.x**2 + ped.position.y**2)]]
		nearest_index = np.argmin(np.array(ped_positions).T[2])
		self.nearest_pedestrian = ped_positions[nearest_index][0:2]

	def ego_pose_cb(self, data):
		quaternion = (data.orientation.x,
					  data.orientation.y,
					  data.orientation.z,
					  data.orientation.w)
		self.current_heading = tf.transformations.euler_from_quaternion(quaternion)[2]

	def check_reachability(self, state):
		x, y, v, h = state[0], state[1], state[2], state[3]
		vel_index = np.argmin(abs(self.velocity_list-v))
		point = Point([x,y])

		critical = False
		ctime_index = 0
		while not critical and ctime_index<self.criticaltime_list.shape[0]:
			if self.critical_polygon[vel_index][ctime_index].contains(point):
				critical = True
			ctime_index += 1

		imminent = False
		itime_index = 0
		while not imminent and itime_index<self.imminenttime_list.shape[0]:
			if self.imminent_polygon[vel_index][itime_index].contains(point):
				imminent = True
			itime_index += 1

		return vel_index, ctime_index-1, itime_index-1


	def init_reachability(self, min_vel=0, max_vel=20, res_vel=0.4, min_ct=0, 
						  max_ct=2, res_ct=0.01, min_it=0, max_it=0.27, res_it=0.01):
		self.velocity_list = np.arange(min_vel,max_vel+res_vel, res_vel)
		self.criticaltime_list = np.arange(min_ct, max_ct+res_ct, res_ct)
		self.imminenttime_list = np.arange(min_it, max_it+res_it, res_it)
		self.critical_reachability = []
		for v in self.velocity_list:
			self.critical_reachability += [[]]
			for ct in self.criticaltime_list:
				self.critical_reachability[-1] += [[[],[]]]
		self.imminent_reachability = []
		for v in self.velocity_list:
			self.imminent_reachability += [[]]
			for it in self.imminenttime_list:
				self.imminent_reachability[-1] += [[[],[]]]
		

		csvfiles = glob.glob(self.csvpath + "*.csv")
		print csvfiles
		for csvfile in csvfiles:
			filename = csvfile.split("/")[-1].split("_")[-1]
			timestamp = float(filename[0:5])
			case = filename[5]
			coordinate = filename[6]
			with open (csvfile, "rb") as csvf:
				rows = csv.reader(csvf)
				for row in rows:
					vel = float(row[0])
					vel_index = np.argmin(abs(self.velocity_list-vel))
					if case=="C" and coordinate=="X":
						time_index = np.argmin(abs(self.criticaltime_list-timestamp))
						for p in row[1:]:
							self.critical_reachability[vel_index][time_index][0] += [float(p)]
					if case=="C" and coordinate=="Y":
						time_index = np.argmin(abs(self.criticaltime_list-timestamp))
						for p in row[1:]:
							self.critical_reachability[vel_index][time_index][1] += [float(p)]
					if case=="I" and coordinate=="X":
						time_index = np.argmin(abs(self.imminenttime_list-timestamp))
						for p in row[1:]:
							self.imminent_reachability[vel_index][time_index][0] += [float(p)]
					if case=="I" and coordinate=="Y":
						time_index = np.argmin(abs(self.imminenttime_list-timestamp))
						for p in row[1:]:
							self.imminent_reachability[vel_index][time_index][1] += [float(p)]
		
		self.critical_polygon = []
		for vi, v in enumerate(self.velocity_list):
			self.critical_polygon += [[]]
			for cti, ct in enumerate(self.criticaltime_list):
				self.critical_polygon[-1] += [Polygon(np.array(self.critical_reachability[vi][cti]).T)]

		self.imminent_polygon = []
		for vi, v in enumerate(self.velocity_list):
			self.imminent_polygon += [[]]
			for iti, it in enumerate(self.imminenttime_list):
				self.imminent_polygon[-1] += [Polygon(np.array(self.imminent_reachability[vi][iti]).T)]

		print ("reachability loaded")


if __name__ == "__main__":

	pcam(csvpath = rospy.get_param('csvpath'))
