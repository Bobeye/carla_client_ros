#!/usr/bin/env python
# Bowen, Dec, 2017
"""Basic CARLA client on ROS."""



from __future__ import print_function
# General Imports
import numpy as np
import random
import time
import sys
import argparse
import logging
import os
import cv2
import matplotlib.pyplot as plt

# Carla imports, Everything can be imported directly from "carla" package
from carla import CARLA
from carla import Control
from carla import Measurements

# ROS Imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3, Accel
from std_msgs.msg import Float64, String, Header
import tf
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from carla_ros_msgs.msg import TrafficLight, Pedestrian, Vehicle, TrafficLights, Pedestrians, Vehicles

from config import *

# Constant that set how offten the episodes are reseted
RESET_FREQUENCY = 100
# ROS node frequency
ROS_FREQUENCY = 5


class CarlaClient():

	def __init__(self, ini_file="CarlaSettings.ini", host="127.0.0.1", port=2000, map="Town02"):
		self.ini_file = ini_file
		self.host = host
		self.port = port
		if map == "Town01":
			self.map = Town01()
		if map == "Town02":
			self.map = Town02()


		# initialize variables
		self.image_rgb = None 
		self.image_depth = None
		self.image_seg = None
		self.ego_point = None
		self.ego_quaternion = None 
		self.ego_accel = None
		self.ego_speed = None
		self.trafficlights = None 
		self.vehicles = None 
		self.pedestrians = None
		self.traffics_array = None
		self.traffics_count = 0

		rospy.init_node('carla_node', anonymous=True)

		# Publisher Setup
		self.image_rgb_pub = rospy.Publisher('/carla/image_rgb', Image, queue_size=1)
		self.image_depth_pub = rospy.Publisher('/carla/image_depth', Image, queue_size=1)
		self.image_seg_pub = rospy.Publisher('/carla/image_seg', Image, queue_size=1)
		self.ego_pose_pub = rospy.Publisher('/carla/ego_pose', Pose, queue_size=1)
		self.ego_accel_pub = rospy.Publisher('/carla/ego_accel', Twist, queue_size=1)
		self.ego_speed_pub = rospy.Publisher('/carla/ego_speed', Float64, queue_size=1)
		self.traffilights_pub = rospy.Publisher('/carla/traffic_lights', TrafficLights, queue_size=1)
		self.pedestrians_pub = rospy.Publisher('/carla/pedestrians', Pedestrians, queue_size=1)
		self.vehicles_pub = rospy.Publisher('/carla/vehicles', Vehicles, queue_size=1)
		self.traffics_markers_pub = rospy.Publisher('/carla/traffics_markers', MarkerArray, queue_size=1)

		self.main_loop()


	def main_loop(self):
		# Carla Setup
		carla =CARLA(self.host,self.port)
		positions = carla.loadConfigurationFile(self.ini_file)
		carla.newEpisode(0)
		capture = time.time()
		i = 1
		# Iterator that will go over the positions on the map after each episode
		iterator_start_positions = 1

		rate = rospy.Rate(ROS_FREQUENCY)
		while not rospy.is_shutdown():
			try:
				"""
					User get the measurements dictionary from the server. 
					Measurements contains:
					* WallTime: Current time on Wall from server machine.
					* GameTime: Current time on Game. Restarts at every episode
					* PlayerMeasurements : All information and events that happens to player
					* Agents : All non-player agents present on the map information such as cars positions, traffic light states
					* BRGA : BGRA optical images
					* Depth : Depth Images
					* Labels : Images containing the semantic segmentation. NOTE: the semantic segmentation must be
						previously activated on the server. See documentation for that. 

				"""
				measurements = carla.getMeasurements()
				self.measurements_process(measurements)
				self.measurements_publish()
				"""
					Sends a control command to the server
					This control structue contains the following fields:
					* throttle : goes from 0 to 1
					* steer : goes from -1 to 1
					* brake : goes from 0 to 1
					* hand_brake : Activate or desactivate the Hand Brake.
					* reverse: Activate or desactive the reverse gear.

				"""

				control = Control()
				control.throttle = 0.9
				control.steer = 0
				carla.sendCommand(control)
				rate.sleep()
						
				
				i+=1


				if i % RESET_FREQUENCY ==0:
						
					print ('Fps for this episode : ',(1.0/((time.time() -capture)/100.0)))
					
					""" 
						Starts another new episode, the episode will have the same configuration as the previous
						one. In order to change configuration, the loadConfigurationFile could be called at any
						time.
					"""
					if iterator_start_positions < len(positions):
						carla.newEpisode(iterator_start_positions)
						iterator_start_positions+=1
					else :
						carla.newEpisode(0)
						iterator_start_positions = 1

					print("Now Starting on Position: ",iterator_start_positions-1)
					capture = time.time()


			except KeyboardInterrupt:
				pass

	def measurements_process(self, measurements):
		img_vec = measurements['BGRA']
		depth_vec = measurements['Depth']
		labels_vec = measurements['Labels']

		# get rgb image
		if len(img_vec)>0:
			img_vec[0] = img_vec[0][:,:,:3]
			img_vec[0] = img_vec[0][:,:,::-1]
			self.image_rgb = np.swapaxes(np.transpose(img_vec[0], (1,0,2)), 0, 1)
			
		
		# get depth image
		if len(depth_vec)>0:
			depth_vec[0] = depth_vec[0][:,:,:3]
			depth_vec[0] = depth_vec[0][:,:,::-1]
			depth_vec[0] = self.convert_depth(depth_vec[0])
			self.image_depth = np.swapaxes(np.transpose(depth_vec[0], (1,0,2)), 0, 1)
			self.image_depth =  cv2.normalize(self.image_depth, (0,255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
			self.image_depth = np.array(self.image_depth, dtype=np.uint8)
			
		# get segmentation image
		if len(labels_vec)>0:
			labels_vec[0] = self.join_classes(labels_vec[0][:,:,2])
			self.image_seg = np.swapaxes(np.transpose(labels_vec[0], (1,0,2)), 0, 1)
			self.image_seg = cv2.normalize(self.image_seg, (0,255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
			self.image_seg = np.array(self.image_seg, dtype=np.uint8)

		# get traffics markers prepared
		self.traffics_array = MarkerArray()

		# ego vehicle states
		ego_x = measurements['PlayerMeasurements'].transform.location.x
		ego_y = measurements['PlayerMeasurements'].transform.location.y
		ego_z = measurements['PlayerMeasurements'].transform.location.z
		ego_ox = measurements['PlayerMeasurements'].transform.orientation.x
		ego_oy = measurements['PlayerMeasurements'].transform.orientation.y
		ego_oz = measurements['PlayerMeasurements'].transform.orientation.z
		ego_ax = measurements['PlayerMeasurements'].acceleration.x
		ego_ay = measurements['PlayerMeasurements'].acceleration.y
		ego_az = measurements['PlayerMeasurements'].acceleration.z
		ego_v = measurements['PlayerMeasurements'].forward_speed
		self.ego_point = Point()
		self.ego_point.x = ego_x
		self.ego_point.y = ego_y
		self.ego_point.z = ego_z
		e2_quaternion = tf.transformations.quaternion_from_euler(ego_ox, ego_oy, ego_oz) # roll pitch yall
		self.ego_quaternion = Quaternion()
		self.ego_quaternion.x = e2_quaternion[0]
		self.ego_quaternion.y = e2_quaternion[1]
		self.ego_quaternion.z = e2_quaternion[2]
		self.ego_quaternion.w = e2_quaternion[3]
		self.ego_accel = Vector3()
		self.ego_accel.x = ego_ax
		self.ego_accel.y = ego_ay
		self.ego_accel.z = ego_az
		self.ego_speed = ego_v
		self.create_ego_marker(ego_x, ego_y, ego_z)



		# get traffics states
		self.vehicles = Vehicles()
		self.pedestrians = Pedestrians()
		self.trafficlights = TrafficLights()
		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = 'velodyne'
		self.vehicles.header = header
		self.pedestrians.header = header
		self.trafficlights.header = header
		for agent in measurements['Agents']:
			if agent.HasField('vehicle'):
				veh = self.carla2rosvehicle(agent)
				self.vehicles.vehicles.append(veh)
				self.create_vehicle_marker(agent.vehicle.transform.location.x-ego_x,
										   agent.vehicle.transform.location.y-ego_y,
										   agent.vehicle.transform.location.z-ego_z)
				
			elif agent.HasField('pedestrian'):
				ped = self.carla2rospedestrian(agent)
				self.pedestrians.pedestrians.append(ped)
				self.create_pedestrian_marker(agent.pedestrian.transform.location.x-ego_x,
											  agent.pedestrian.transform.location.y-ego_y,
											  agent.pedestrian.transform.location.z-ego_z)

			elif agent.HasField('traffic_light'): # green=0, yellow=1, red=2
				if agent.traffic_light.state == 2:
					tlt = self.carla2rostrafficlight(agent)
					self.trafficlights.trafficLights.append(tlt)

			elif agent.HasField('speed_limit_sign'):
				pass

	def create_ego_marker(self, x, y, z):
		marker = Marker()
		marker.id = self.traffics_count
		marker.type = marker.SPHERE
		marker.header.frame_id = "velodyne"
		marker.action = marker.ADD
		marker.scale.x = 1.
		marker.scale.y = 1.
		marker.scale.z = 1.
		marker.color.r = 0.1
		marker.color.g = 0.0
		marker.color.b = 1.0
		marker.color.a = 1.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = 0
		marker.pose.position.y = 0
		marker.pose.position.z = 0
		self.traffics_array.markers.append(marker)
		self.traffics_count += 1

	def create_vehicle_marker(self, x, y, z):
		mx, my, mz = self.world2map(x, y, z)
		marker = Marker()
		marker.id = self.traffics_count
		marker.type = marker.CUBE
		marker.header.frame_id = "velodyne"
		marker.action = marker.ADD
		marker.scale.x = 1.
		marker.scale.y = 1.
		marker.scale.z = 1.
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.color.a = 1.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = mx
		marker.pose.position.y = my
		marker.pose.position.z = mz
		self.traffics_array.markers.append(marker)
		self.traffics_count += 1

	def create_pedestrian_marker(self, x, y, z):
		mx, my, mz = self.world2map(x, y, z)
		marker = Marker()
		marker.id = self.traffics_count
		marker.type = marker.CUBE
		marker.header.frame_id = "velodyne"
		marker.action = marker.ADD
		marker.scale.x = 1.
		marker.scale.y = 1.
		marker.scale.z = 1.
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = mx
		marker.pose.position.y = my
		marker.pose.position.z = mz
		self.traffics_array.markers.append(marker)
		self.traffics_count += 1


	def world2map(self, x,y,z):
		map_x = (x+self.map.world2map_T[0]-self.map.offset[0])/self.map.density/10.
		map_y = (y+self.map.world2map_T[1]-self.map.offset[1])/self.map.density/10.
		map_z = (z+self.map.world2map_T[2]-self.map.offset[2])/self.map.density
		return map_x, map_y, map_z

	def measurements_publish(self):
		if self.image_rgb is not None:
			image_rgb_frame = CvBridge().cv2_to_imgmsg(self.image_rgb, "rgb8")
			self.image_rgb_pub.publish(image_rgb_frame)
		
		if self.image_depth is not None:
			image_depth_frame = CvBridge().cv2_to_imgmsg(self.image_depth, "rgb8")
			self.image_depth_pub.publish(image_depth_frame)

		if self.image_seg is not None:
			image_seg_frame = CvBridge().cv2_to_imgmsg(self.image_seg, "rgb8")
			self.image_seg_pub.publish(image_seg_frame)

		if self.ego_point is not None and self.ego_quaternion is not None:
			ego_pose_frame = Pose()
			ego_pose_frame.position = self.ego_point
			ego_pose_frame.orientation = self.ego_quaternion
			self.ego_pose_pub.publish(ego_pose_frame)

		if self.ego_accel is not None:
			ego_accel_frame = Twist()
			ego_accel_frame.linear = self.ego_accel
			self.ego_accel_pub.publish(ego_accel_frame)

		if self.ego_speed is not None:
			ego_speed_frame = Float64()
			ego_speed_frame.data = self.ego_speed
			self.ego_speed_pub.publish(ego_speed_frame)

		if self.vehicles is not None:
			self.vehicles_pub.publish(self.vehicles)
			self.vehicles = None

		if self.pedestrians is not None:
			self.pedestrians_pub.publish(self.pedestrians)
			self.pedestrians = None

		if self.trafficlights is not None:
			self.traffilights_pub.publish(self.trafficlights)
			self.trafficlights = None
			
		if self.traffics_array is not None:
			self.traffics_markers_pub.publish(self.traffics_array)
			self.traffics_array = None
			self.traffics_count = 0

	def carla2rosvehicle(self, agent):
		veh = Vehicle()
		
		veh_pos = Point()
		veh_pos.x = agent.vehicle.transform.location.x
		veh_pos.y = agent.vehicle.transform.location.y
		veh_pos.z = agent.vehicle.transform.location.z
		veh.position = veh_pos

		return veh

	def carla2rospedestrian(self, agent):
		ped = Pedestrian()
		
		ped_pos = Point()
		ped_pos.x = agent.pedestrian.transform.location.x
		ped_pos.y = agent.pedestrian.transform.location.y
		ped_pos.z = agent.pedestrian.transform.location.z
		ped.position = ped_pos

		return ped

	def carla2rostrafficlight(self, agent):
		tlt = TrafficLight()

		tlt_pos = Point()
		tlt_pos.x = agent.traffic_light.transform.location.x
		tlt_pos.y = agent.traffic_light.transform.location.y
		tlt_pos.z = agent.traffic_light.transform.location.z
		tlt.position = tlt_pos

		return tlt

	# Function for making colormaps
	def grayscale_colormap(self, img,colormap):
		cmap = plt.get_cmap(colormap)
		rgba_img = cmap(img)
		rgb_img = np.delete(rgba_img, 3, 2)
		return rgb_img

	# Function to convert depth to human readable format 
	def convert_depth(self, depth):
		depth = depth.astype(np.float32)
		gray_depth = ((depth[:,:,0] +  depth[:,:,1]*256.0 +  depth[:,:,2]*256.0*256.0)/(256.0*256.0*256.0 ))
		color_depth =self.grayscale_colormap(gray_depth,'jet')*255
		return color_depth

	def join_classes(self, labels_image):
		classes_join = {0:[0,0,0],1:[64,64,64],2:[96,96,96],3:[255,255,255],5:[128,128,128],12:2,9:[0,255,0],\
		11:[32,32,32],4:[255,0,0],10:[0,0,255],8:[255,0,255],6:[196,196,196],7:[128,0,128]}

		compressed_labels_image = np.zeros((labels_image.shape[0],labels_image.shape[1],3)) 
		for key,value in classes_join.items():
			compressed_labels_image[np.where(labels_image==key)] = value

		return compressed_labels_image
			




if __name__ == "__main__":
	CarlaClient(ini_file=rospy.get_param('ini'))

