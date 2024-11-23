#!/usr/bin/env python3
import os
import time
from typing import Optional

import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, EpisodeStart
from nav_msgs.msg import Odometry

from visual_lane_servoing.include import odometry

class EncoderPoseNode(DTROS):
    """
    Computes an estimate of the Duckiebot pose using the wheel encoders.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~encoder_localization (:obj:`PoseStamped`): The computed position
    Subscribers:
        ~/left_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
        ~/right_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
    """

    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(EncoderPoseNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.log("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")

        # internal state
        # - odometry
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0
        self.x_curr = 0.0
        self.y_curr = 0.0
        self.theta_curr = 0.0

        # - PID controller
        #   - previous tracking error, starts at 0
        self.prev_e = 0.0
        #   - previous tracking error integral, starts at 0
        self.prev_int = 0.0
        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.v_0 = 0.0
        # reference y for PID lateral control activity - zero so can set interactively at runtime
        self.y_ref = 0.0

        # initial reference signal for heading control activity
        self.theta_ref = np.deg2rad(0.0)
        # initializing omega command to the robot
        self.omega = 0.0

        # Init the parameters
        self.resetParameters()

        # nominal R and L, you may change these if needed:

        self.R = 0.0318  # meters, default value of wheel radius
        self.baseline = 0.1  # meters, default value of baseline

        # Defining subscribers:

        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_driver_node/tick"
        rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.cbLeftEncoder, queue_size=1)

        # Wheel encoder subscriber:
        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_driver_node/tick"
        rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.cbRightEncoder, queue_size=1)

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/encoder_localization", Odometry, queue_size=1, dt_topic_type=TopicType.LOCALIZATION
        )


        # Wait until the encoders data is received, then start the controller
        self.duckiebot_is_moving = False
        self.STOP = False

        # For encoders syncronization:
        self.RIGHT_RECEIVED = False
        self.LEFT_RECEIVED = False

        self.log("Initialized.")

    def resetParameters(self):
        # Add the node parameters to the parameters dictionary
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        # Initializing the odometry
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0

        self.x_curr = 0.0
        self.y_curr = 0.0
        self.theta_curr = 0.0

        # Initializing the PID controller parameters
        self.prev_e = 0.0  # previous tracking error, starts at 0
        self.prev_int = 0.0  # previous tracking error integral, starts at 0
        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.v_0 = 0.0
        # reference y for PID lateral control activity - zero so can set interactively at runtime
        self.y_ref = 0.0


    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        # initializing ticks to stored absolute value
        if self.left_tick_prev is None:
            ticks = encoder_msg.data
            self.left_tick_prev = ticks
            return

        # running the DeltaPhi() function copied from the notebooks to calculate rotations
        delta_phi_left, self.left_tick_prev = odometry.delta_phi(
            encoder_msg.data, self.left_tick_prev, encoder_msg.resolution
        )
        self.delta_phi_left += delta_phi_left

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.LEFT_RECEIVED = True
        self.posePublisher()

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        if self.right_tick_prev is None:
            ticks = encoder_msg.data
            self.right_tick_prev = ticks
            return

        # calculate rotation of right wheel
        delta_phi_right, self.right_tick_prev = odometry.delta_phi(
            encoder_msg.data, self.right_tick_prev, encoder_msg.resolution
        )
        self.delta_phi_right += delta_phi_right

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.RIGHT_RECEIVED = True
        self.posePublisher()

    def posePublisher(self):
        """
        Publish the pose of the Duckiebot given by the kinematic model
            using the encoders.
        Publish:
            ~/pose (:obj:`PoseStamped`): Duckiebot pose.
        """
        if self.STOP or not (self.LEFT_RECEIVED and self.RIGHT_RECEIVED):
            return

        # synch incoming messages from encoders
        self.LEFT_RECEIVED = self.RIGHT_RECEIVED = False

        self.x_curr, self.y_curr, theta_curr = odometry.estimate_pose(
            self.R,
            self.baseline,
            self.x_prev,
            self.y_prev,
            self.theta_prev,
            self.delta_phi_left,
            self.delta_phi_right,
        )

        self.theta_curr = self.angle_clamp(theta_curr)  # angle always between 0,2pi

        # self.loging to screen for debugging purposes
        self.log("              ODOMETRY             ")
        # self.log(f"Baseline : {self.baseline}   R: {self.R}")
        self.log(f"Theta : {np.rad2deg(self.theta_curr)} deg,  x: {self.x_curr} m,  y: {self.y_curr} m")
        self.log(
            f"Rotation left wheel : {np.rad2deg(self.delta_phi_left)} deg,   "
            f"Rotation right wheel : {np.rad2deg(self.delta_phi_right)} deg"
        )
        self.log(f"Prev Ticks left : {self.left_tick_prev}   Prev Ticks right : {self.right_tick_prev}")
        # self.log(
        #     f"Prev integral error : {self.prev_int}")

        self.duckiebot_is_moving = abs(self.delta_phi_left) > 0 or abs(self.delta_phi_right) > 0

        # Calculate new odometry only when new data from encoders arrives
        self.delta_phi_left = self.delta_phi_right = 0

        # Current estimate becomes previous estimate at next iteration
        self.x_prev = self.x_curr
        self.y_prev = self.y_curr
        self.theta_prev = self.theta_curr

        # Creating message to plot pose in RVIZ
        odom = Odometry()
        odom.header.frame_id = "map"
        odom.header.stamp = rospy.Time.now()

        odom.pose.pose.position.x = self.x_curr  # x position - estimate
        odom.pose.pose.position.y = self.y_curr  # y position - estimate
        odom.pose.pose.position.z = 0  # z position - no flying allowed in Duckietown

        # these are quaternions - stuff for a different course!
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = np.sin(self.theta_curr / 2)
        odom.pose.pose.orientation.w = np.cos(self.theta_curr / 2)

        self.db_estimated_pose.publish(odom)



    def onShutdown(self):
        super(EncoderPoseNode, self).on_shutdown()

    @staticmethod
    def angle_clamp(theta):
        if theta > 2 * np.pi:
            return theta - 2 * np.pi
        elif theta < -2 * np.pi:
            return theta + 2 * np.pi
        else:
            return theta


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = EncoderPoseNode(node_name="encoder_pose_node")
    # Keep it spinning
    rospy.spin()