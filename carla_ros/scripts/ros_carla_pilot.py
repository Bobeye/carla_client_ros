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


		rospy.init_node('carla_pilot_node', anonymous=True)

		# Subscriber Setup
		rospy.Subscriber('/carla/image_rgb', Image, self.image_rgb_cb)
		rospy.Subscriber('/carla/ego_pose', Pose)
		rospy.Subscriber('/carla/ego_accel', Twist)
		rospy.Subscriber('/carla/ego_speed', Float64)

		# Publisher Setup





