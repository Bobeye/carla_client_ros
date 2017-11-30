#!/usr/bin/env python
# Bowen, Dec, 2017
"""Autopilot for CARLA."""



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

# ROS Imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3, Accel
from std_msgs.msg import Float64, String, Header
import tf
from geometry_msgs.msg import TransformStamped

from carla_ros_msgs.msg import TrafficLight, Pedestrian, Vehicle, TrafficLights, Pedestrians, Vehicles

# ROS node frequency
ROS_FREQUENCY = 5

class CarlaPilot():


	def __init__(self):

		self.current_speed = None
		self.current_pose = None

		rospy.init_node('carla_pilot_node', anonymous=True)

		# Subscriber Setup
		rospy.Subscriber('/carla/image_rgb', Image, self.image_rgb_cb)
		rospy.Subscriber('/carla/ego_pose', Pose, self.ego_pose_cb)
		rospy.Subscriber('/carla/ego_accel', Twist, self.ego_accel_cb)
		rospy.Subscriber('/carla/ego_speed', Float64, self.ego_speed_cb)

		# Publisher Setup
		self.throttle_pub = rospy.Publisher('/carla/throttle_command', Float64, queue_size=1)
		self.brake_pub = rospy.Publisher('/carla/brake_command', Float64, queue_size=1)
		self.steering_pub = rospy.Publisher('/carla/steer_command', Float64, queue_size=1)

		self.main_loop()

	def main_loop(self):
		rate = rospy.Rate(ROS_FREQUENCY)
		while not rospy.is_shutdown():

			if self.current_pose is not None:
				print (self.current_pose)


			rate.sleep()


	def image_rgb_cb(self, data):
		self.image_frame = data


	def ego_pose_cb(self, data):
		ego_x = data.position.x 
		ego_y = data.position.y 
		ego_z = data.position.z 
		self.current_pose = [ego_x, ego_y, ego_z]


	def ego_accel_cb(self, data):
		self.ego_accel_fram = data

	def ego_speed_cb(self, data):
		self.current_speed = data.data


if __name__ == "__main__":
	CarlaPilot()

