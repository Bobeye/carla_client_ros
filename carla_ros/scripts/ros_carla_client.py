#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# ROS support for CARLA client

# By Bowen, Dec, 2017 

"""Basic CARLA client in ROS."""

from __future__ import print_function

import argparse
import logging
import random
import sys
import time
import numpy as np
import cv2

from carla.client import make_carla_client
from carla import image_converter
from carla import sensor
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

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

# Constant that set how offten the episodes are reseted
RESET_FREQUENCY = 100
# ROS node frequency
ROS_FREQUENCY = 10

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180


class CarlaClient():


	def __init__(self, host="localhost", port=2000, autopilot_on=True, 
				 settings_filepath=None, number_of_episodes=100, frames_per_episode=300,
				 city_name="Town01"):
		self.host = host
		self.port = port
		self.autopilot_on = autopilot_on
		self.settings_filepath = settings_filepath
		self.number_of_episodes = number_of_episodes
		self.frames_per_episode = frames_per_episode
		self.city_name = city_name

		# initialize variables
		self.image_rgb = None 
		self.image_depth = None
		self.image_seg = None
		self.traffics_array = None
		self.traffics_count = 0

		self.map = CarlaMap(city_name) if city_name is not None else None
		self.map_shape = self.map.map_image.shape if city_name is not None else None
		self.map_view = self.map.get_map(WINDOW_HEIGHT) if city_name is not None else None
		self.image_biv = np.transpose(np.array(self.map_view[:, :, :3].swapaxes(0,1), dtype=np.uint8), (1,0,2))


		rospy.init_node('carla_node', anonymous=True)

		# Publisher Setup
		self.image_rgb_pub = rospy.Publisher('/carla/image_rgb', Image, queue_size=1)
		self.image_depth_pub = rospy.Publisher('/carla/image_depth', Image, queue_size=1)
		self.image_seg_pub = rospy.Publisher('/carla/image_seg', Image, queue_size=1)
		self.image_biv_pub = rospy.Publisher('/carla/image_biv', Image, queue_size=1)
		self.traffics_markers_pub = rospy.Publisher('/carla/traffics_markers', MarkerArray, queue_size=1)

		self.main_loop()

	def main_loop(self):
		with make_carla_client(self.host, self.port) as client:
			print('CarlaClient connected')
			try:
				while not rospy.is_shutdown():
					for episode in range(0, self.number_of_episodes):
						settings = self.make_carla_settings()
						# Now we load these settings into the server. The server replies
						# with a scene description containing the available start spots for
						# the player. Here we can provide a CarlaSettings object or a
						# CarlaSettings.ini file as string.
						scene = client.load_settings(settings)
						# Choose one player start at random.
						number_of_player_starts = len(scene.player_start_spots)
						player_start = random.randint(0, max(0, number_of_player_starts - 1))
						# Notify the server that we want to start the episode at the
						# player_start index. This function blocks until the server is ready
						# to start the episode.
						print('Starting new episode...')
						client.start_episode(player_start)

						rate = rospy.Rate(ROS_FREQUENCY)
					
						# Iterate every frame in the episode.
						for frame in range(0, self.frames_per_episode):

							# Read the data produced by the server this frame.
							measurements, sensor_data = client.read_data()
							self.measurements_process(measurements)
							self.sensor_process(sensor_data)

							if not self.autopilot_on:
								client.send_control(
									steer=random.uniform(-1.0, 1.0),
									throttle=0.5,
									brake=False,
									hand_brake=False,
									reverse=False)
							else:
								# Together with the measurements, the server has sent the
								# control that the in-game autopilot would do this frame. We
								# can enable autopilot by sending back this control to the
								# server. We can modify it if wanted, here for instance we
								# will add some noise to the steer.

								control = measurements.player_measurements.autopilot_control
								control.steer += random.uniform(-0.1, 0.1)
								client.send_control(control)
							rate.sleep()

			except KeyboardInterrupt:
				pass

	def measurements_process(self, measurements):
		self.traffics_array = MarkerArray()

		# Function to get car position on map.
		ego_position = self.map.get_position_on_map([
			measurements.player_measurements.transform.location.x,
			measurements.player_measurements.transform.location.y,
			measurements.player_measurements.transform.location.z])
		# Function to get orientation of the road car is in.
		ego_orientation = self.map.get_lane_orientation([
			measurements.player_measurements.transform.location.x,
			measurements.player_measurements.transform.location.y,
			measurements.player_measurements.transform.location.z])
		# publish ego marker
		self.create_marker(0, 0, 0, shape=2, cr=1.0, cg=0., cb=0., marker_scale=3.0)
		# mark on the biv image
		new_window_width =(float(WINDOW_HEIGHT)/float(self.map_shape[0]))*float(self.map_shape[1])
		w_pos = int(ego_position[0]*(float(WINDOW_HEIGHT)/float(self.map_shape[0])))
		h_pos =int(ego_position[1] *(new_window_width/float(self.map_shape[1])))


		agent_positions = measurements.non_player_agents
		for agent in agent_positions:
			if agent.HasField('vehicle'):
				agent_position = self.map.get_position_on_map([
					agent.vehicle.transform.location.x,
					agent.vehicle.transform.location.y,
					agent.vehicle.transform.location.z])
				self.create_marker(agent_position[0]-ego_position[0], 
								   agent_position[1]-ego_position[1], 
								   0, 
								   shape=1, cr=0.0, cg=1., cb=0., marker_scale=2.0)
				
			if agent.HasField('pedestrian'):
				agent_position = self.map.get_position_on_map([
					agent.pedestrian.transform.location.x,
					agent.pedestrian.transform.location.y,
					agent.pedestrian.transform.location.z])
				self.create_marker(agent_position[0]-ego_position[0], 
								   agent_position[1]-ego_position[1], 
								   0, 
								   shape=1, cr=0.0, cg=1., cb=1.)
		self.traffics_markers_pub.publish(self.traffics_array)
		self.traffics_array = None
		self.traffics_count = 0

		if self.image_biv is not None:
			image_biv_frame = CvBridge().cv2_to_imgmsg(cv2.circle(np.copy(self.image_biv),(w_pos,h_pos), 8, (0,0,255), -1), "rgb8")
			self.image_biv_pub.publish(image_biv_frame)
	
	def create_marker(self, x, y, z, shape=0, cr=None, cg=None, cb=None, dist_scale=0.1, marker_scale=1.0):
		marker = Marker()
		marker.id = self.traffics_count
		marker.type = shape
		marker.header.frame_id = "velodyne"
		marker.action = marker.ADD
		marker.scale.x = 1.* marker_scale
		marker.scale.y = 1.* marker_scale
		marker.scale.z = 1.* marker_scale
		marker.color.r = cr
		marker.color.g = cg
		marker.color.b = cb
		marker.color.a = 1.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = x*dist_scale
		marker.pose.position.y = y*dist_scale
		marker.pose.position.z = z*dist_scale
		self.traffics_array.markers.append(marker)
		self.traffics_count += 1

	def sensor_process(self, sensor_data):
		self.image_rgb = np.transpose(image_converter.to_rgb_array(sensor_data['CameraRGB']).swapaxes(0, 1), (1,0,2))
		self.image_depth = np.transpose(image_converter.depth_to_logarithmic_grayscale(sensor_data['CameraDepth']).swapaxes(0, 1), (1,0,2))
		self.image_depth = np.array(self.image_depth, dtype=np.uint8)
		self.image_seg = np.transpose(image_converter.labels_to_cityscapes_palette(sensor_data['CameraSemSeg']).swapaxes(0, 1), (1,0,2))
		self.image_seg = np.array(self.image_seg, dtype=np.uint8)

		if self.image_rgb is not None:
			image_rgb_frame = CvBridge().cv2_to_imgmsg(self.image_rgb, "rgb8")
			self.image_rgb_pub.publish(image_rgb_frame)
		
		if self.image_depth is not None:
			image_depth_frame = CvBridge().cv2_to_imgmsg(self.image_depth, "rgb8")
			self.image_depth_pub.publish(image_depth_frame)

		if self.image_seg is not None:
			image_seg_frame = CvBridge().cv2_to_imgmsg(self.image_seg, "rgb8")
			self.image_seg_pub.publish(image_seg_frame)

	def make_carla_settings(self):

		if self.settings_filepath is None:
			"""Make a CarlaSettings object with the settings we need."""
			settings = CarlaSettings()
			settings.set(
				SynchronousMode=True,
				SendNonPlayerAgentsInfo=True,
				NumberOfVehicles=20,
				NumberOfPedestrians=40,
				WeatherId=random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
			settings.randomize_seeds()
			camera0 = sensor.Camera('CameraRGB')
			camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
			camera0.set_position(200, 0, 140)
			camera0.set_rotation(0.0, 0.0, 0.0)
			settings.add_sensor(camera0)
			camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
			camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
			camera1.set_position(200, 0, 140)
			camera1.set_rotation(0.0, 0.0, 0.0)
			settings.add_sensor(camera1)
			camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
			camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
			camera2.set_position(200, 0, 140)
			camera2.set_rotation(0.0, 0.0, 0.0)
			settings.add_sensor(camera2)
		else:
			# Alternatively, we can load these settings from a file.
			with open(self.settings_filepath, 'r') as fp:
				settings = fp.read()

		return settings

if __name__ == "__main__":
	CarlaClient()



#     def run_carla_client():

#         # We assume the CARLA server is already waiting for a client to connect at
#         # host:port. To create a connection we can use the `make_carla_client`
#         # context manager, it creates a CARLA client object and starts the
#         # connection. It will throw an exception if something goes wrong. The
#         # context manager makes sure the connection is always cleaned up on exit.
#         with make_carla_client(host, port) as client:
#             print('CarlaClient connected')

#             for episode in range(0, number_of_episodes):
#                 # Start a new episode.

#                 if settings_filepath is None:

#                     # Create a CarlaSettings object. This object is a wrapper around
#                     # the CarlaSettings.ini file. Here we set the configuration we
#                     # want for the new episode.
#                     settings = CarlaSettings()
#                     settings.set(
#                         SynchronousMode=True,
#                         SendNonPlayerAgentsInfo=True,
#                         NumberOfVehicles=20,
#                         NumberOfPedestrians=40,
#                         WeatherId=random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
#                     settings.randomize_seeds()

#                     # Now we want to add a couple of cameras to the player vehicle.
#                     # We will collect the images produced by these cameras every
#                     # frame.

#                     # The default camera captures RGB images of the scene.
#                     camera0 = Camera('CameraRGB')
#                     # Set image resolution in pixels.
#                     camera0.set_image_size(800, 600)
#                     # Set its position relative to the car in centimeters.
#                     camera0.set_position(30, 0, 130)
#                     settings.add_sensor(camera0)

#                     # Let's add another camera producing ground-truth depth.
#                     camera1 = Camera('CameraDepth', PostProcessing='Depth')
#                     camera1.set_image_size(800, 600)
#                     camera1.set_position(30, 0, 130)
#                     settings.add_sensor(camera1)

#                 else:

#                     # Alternatively, we can load these settings from a file.
#                     with open(settings_filepath, 'r') as fp:
#                         settings = fp.read()

#                 # Now we load these settings into the server. The server replies
#                 # with a scene description containing the available start spots for
#                 # the player. Here we can provide a CarlaSettings object or a
#                 # CarlaSettings.ini file as string.
#                 scene = client.load_settings(settings)

#                 # Choose one player start at random.
#                 number_of_player_starts = len(scene.player_start_spots)
#                 player_start = random.randint(0, max(0, number_of_player_starts - 1))

#                 # Notify the server that we want to start the episode at the
#                 # player_start index. This function blocks until the server is ready
#                 # to start the episode.
#                 print('Starting new episode...')
#                 client.start_episode(player_start)

#                 # Iterate every frame in the episode.
#                 for frame in range(0, frames_per_episode):

#                     # Read the data produced by the server this frame.
#                     measurements, sensor_data = client.read_data()

#                     # Print some of the measurements.
#                     print_measurements(measurements)

#                     # Save the images to disk if requested.
#                     if save_images_to_disk:
#                         for name, image in sensor_data.items():
#                             image.save_to_disk(image_filename_format.format(episode, name, frame))

#                     # We can access the encoded data of a given image as numpy
#                     # array using its "data" property. For instance, to get the
#                     # depth value (normalized) at pixel X, Y
#                     #
#                     #     depth_array = sensor_data['CameraDepth'].data
#                     #     value_at_pixel = depth_array[Y, X]
#                     #

#                     # Now we have to send the instructions to control the vehicle.
#                     # If we are in synchronous mode the server will pause the
#                     # simulation until we send this control.

#                     if not autopilot_on:

#                         client.send_control(
#                             steer=random.uniform(-1.0, 1.0),
#                             throttle=0.5,
#                             brake=False,
#                             hand_brake=False,
#                             reverse=False)

#                     else:

#                         # Together with the measurements, the server has sent the
#                         # control that the in-game autopilot would do this frame. We
#                         # can enable autopilot by sending back this control to the
#                         # server. We can modify it if wanted, here for instance we
#                         # will add some noise to the steer.

#                         control = measurements.player_measurements.autopilot_control
#                         control.steer += random.uniform(-0.1, 0.1)
#                         client.send_control(control)


# # def print_measurements(measurements):
# #     number_of_agents = len(measurements.non_player_agents)
# #     player_measurements = measurements.player_measurements
# #     message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
# #     message += '{speed:.2f} km/h, '
# #     message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
# #     message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
# #     message += '({agents_num:d} non-player agents in the scene)'
# #     message = message.format(
# #         pos_x=player_measurements.transform.location.x / 100, # cm -> m
# #         pos_y=player_measurements.transform.location.y / 100,
# #         speed=player_measurements.forward_speed,
# #         col_cars=player_measurements.collision_vehicles,
# #         col_ped=player_measurements.collision_pedestrians,
# #         col_other=player_measurements.collision_other,
# #         other_lane=100 * player_measurements.intersection_otherlane,
# #         offroad=100 * player_measurements.intersection_offroad,
# #         agents_num=number_of_agents)
# #     print_over_same_line(message)


# def main():
#     # argparser = argparse.ArgumentParser(description=__doc__)
#     # argparser.add_argument(
#     #     '-v', '--verbose',
#     #     action='store_true',
#     #     dest='debug',
#     #     help='print debug information')
#     # argparser.add_argument(
#     #     '--host',
#     #     metavar='H',
#     #     default='localhost',
#     #     help='IP of the host server (default: localhost)')
#     # argparser.add_argument(
#     #     '-p', '--port',
#     #     metavar='P',
#     #     default=2000,
#     #     type=int,
#     #     help='TCP port to listen to (default: 2000)')
#     # argparser.add_argument(
#     #     '-a', '--autopilot',
#     #     action='store_true',
#     #     help='enable autopilot')
#     # argparser.add_argument(
#     #     '-i', '--images-to-disk',
#     #     action='store_true',
#     #     help='save images to disk')
#     # argparser.add_argument(
#     #     '-c', '--carla-settings',
#     #     metavar='PATH',
#     #     default=None,
#     #     help='Path to a "CarlaSettings.ini" file')

#     # args = argparser.parse_args()

#     # log_level = logging.DEBUG if args.debug else logging.INFO
#     # logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

#     # logging.info('listening to server %s:%s', args.host, args.port)

#     while True:
#         try:

#             run_carla_client(
#                 host='localhost',
#                 port=2000,
#                 autopilot_on=True,
#                 settings_filepath=args.carla_settings)

#             print('Done.')
#             return

#         except TCPConnectionError as error:
#             logging.error(error)
#             time.sleep(1)
#         except Exception as exception:
#             logging.exception(exception)
#             sys.exit(1)


# if __name__ == '__main__':

#     try:
#         main()
#     except KeyboardInterrupt:
#         print('\nCancelled by user. Bye!')
