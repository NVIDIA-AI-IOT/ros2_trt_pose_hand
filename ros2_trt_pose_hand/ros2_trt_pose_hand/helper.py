# ---------------------------------------------------------------------------------------
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------------------
# TRT_pose related
import cv2
import os
import json
import numpy as np
import math
from ros2_trt_pose_hand.utils import preprocess, load_params, load_model, draw_objects, draw_joints, load_image
from ros2_trt_pose_hand.utils import load_svm
from ros2_trt_pose_hand.preprocessdata import preprocessdata
# Gesture Classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# ROS2 related
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from hand_pose_msgs.msg import FingerPoints, HandPoseDetection  # For pose_msgs
from rclpy.duration import Duration
from std_msgs.msg import String

class TRTHandPose(Node):
    def __init__(self):
        super().__init__('trt_pose_hand')
        self.height = 224
        self.width = 224
        self.num_parts = None
        self.num_links = None
        self.model_weights = None
        self.parse_objects = None
        self.counts = None
        self.peaks = None
        self.objects = None
        self.topology = None
        self.hand_pose_skeleton = None
        self.model = None
        self.clf = None # SVM Model
        self.preprocessdata = None
        self.image = None
        self.gesture_type = None
        # ROS2 parameters
        self.declare_parameter('base_dir', os.getenv("HOME") + 'gesture_models')
        # Based Dir should contain: model_file resnet/densenet, human_pose json file
        self.declare_parameter('point_range', 10)  # default range is 0 to 10
        self.declare_parameter('show_image', True)  # Show image in cv2.imshow
        self.declare_parameter('show_gesture', True) # Shows gestures (fist, pan, stop, fine, peace, no hand).
        self.base_dir = self.get_parameter('base_dir')._value
        self.json_file = os.path.join(self.base_dir, 'hand_pose.json') 
        self.point_range = self.get_parameter('point_range')._value
        self.show_image_param = self.get_parameter('show_image')._value
        self.show_gesture_param = self.get_parameter('show_gesture')._value
        # ROS2 related init
        # Image subscriber from cam2image
        self.subscriber_ = self.create_subscription(ImageMsg, 'image', self.read_cam_callback, 10)
        self.image_pub = self.create_publisher(ImageMsg, 'detections_image', 10)
        # Publisher for Body Joints and Skeleton
        self.hand_joints_pub = self.create_publisher(Marker, 'hand_joints', 1000)
        self.hand_skeleton_pub = self.create_publisher(Marker, 'hand_skeleton', 10)
        # Publishing pose Message
        self.publish_pose = self.create_publisher(HandPoseDetection, 'hand_pose_msgs', 100)
        # Publishing gesture classifcation
        self.publish_gesture = self.create_publisher(String, "gesture_class", 100)

    def start(self):
        self.get_logger().info("Loading Parameters\n")
        self.num_parts, self.num_links, self.model_weights, self.parse_objects, self.topology = load_params(base_dir=self.base_dir, hand_pose_json=self.json_file)
        with open(os.path.join(self.base_dir,'gesture.json'), 'r') as f:
            gesture = json.load(f)
            self.gesture_type = gesture["classes"]
        self.get_logger().info("Loading model weights\n")
        self.clf = load_svm(base_dir=self.base_dir)
        self.model = load_model(self.base_dir, self.num_parts, self.num_links, self.model_weights, self.height, self.width)
        self.get_logger().info("Model weights loaded...\n Waiting for images...\n")
        self.preprocessdata = preprocessdata(self.topology, self.num_parts)


    def execute(self):
        data = preprocess(image=self.image, width=self.width, height=self.height)
        cmap, paf = self.model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        self.counts, self.objects, self.peaks = self.parse_objects(cmap, paf)
        joints = self.preprocessdata.joints_inference(self.image, self.counts, self.objects, self.peaks)
        with open(self.json_file, 'r') as f:
            hand_pose = json.load(f)
        hand_pose_skeleton = hand_pose ['skeleton']
        annotated_image = draw_joints(self.image, joints, hand_pose_skeleton)
        # cv2.imwrite('unname_pose.jpg', annotated_image)
        
        self.parse_k()
        print("Show gesture Parameter:{}".format(self.show_gesture_param))
        if self.show_gesture_param:
            self.parse_gesture(joints=joints)
        return annotated_image

    # Borrowed from OpenPose-ROS repo
    def image_np_to_image_msg(self, image_np):
        image_msg = ImageMsg()
        image_msg.height = image_np.shape[0]
        image_msg.width = image_np.shape[1]
        image_msg.encoding = 'bgr8'
        image_msg.data = image_np.tostring()
        image_msg.step = len(image_msg.data) // image_msg.height
        image_msg.header.frame_id = 'map'
        return image_msg

    def init_finger_points_msg(self):
        finger_points = FingerPoints()
        finger_points.x = float('NaN')
        finger_points.y = float('NaN')
        finger_points.confidence = float('NaN')
        return finger_points

    # Subscribe and Publish to image topic
    def read_cam_callback(self, msg):
        img = np.asarray(msg.data)
        self.image = np.reshape(img, (msg.height, msg.width, 3))
        self.annotated_image = self.execute()

        #image_msg = self.image_np_to_image_msg(self.image)
        image_msg = self.image_np_to_image_msg(self.annotated_image)
        self.image_pub.publish(image_msg)
        if self.show_image_param:
            cv2.imshow('frame', self.annotated_image)
            cv2.waitKey(1)

    def write_finger_points_msg(self, pixel_location):
        finger_points = FingerPoints()
        finger_points.y = float(pixel_location[0] * self.height)
        finger_points.x = float(pixel_location[1] * self.width)
        return finger_points

    def parse_gesture(self, joints):
        dist_bn_joints = self.preprocessdata.find_distance(joints)
        gesture = self.clf.predict([dist_bn_joints,[0]*self.num_parts*self.num_parts])
        gesture_joints = gesture[0]
        self.preprocessdata.prev_queue.append(gesture_joints)
        self.preprocessdata.prev_queue.pop(0)
        gesture_label = self.preprocessdata.print_label(self.preprocessdata.prev_queue, self.gesture_type)
        msg = String()
        msg.data = gesture_label
        self.publish_gesture.publish(msg)
        self.get_logger().info('Hand Pose Classified as: "%s"' % msg.data)

    def init_markers_spheres(self):
        marker_joints = Marker()
        marker_joints.header.frame_id = '/map'
        marker_joints.id = 1
        marker_joints.ns = "joints"
        marker_joints.type = marker_joints.SPHERE_LIST
        marker_joints.action = marker_joints.ADD
        marker_joints.scale.x = 0.7
        marker_joints.scale.y = 0.7
        marker_joints.scale.z = 0.7
        marker_joints.color.a = 1.0
        marker_joints.color.r = 1.0
        marker_joints.color.g = 0.0
        marker_joints.color.b = 0.0
        marker_joints.lifetime = Duration(seconds=3, nanoseconds=5e2).to_msg()
        return marker_joints

    def init_markers_lines(self):
        marker_line = Marker()
        marker_line.header.frame_id = '/map'
        marker_line.id = 1
        marker_line.ns = "joint_line"
        marker_line.header.stamp = self.get_clock().now().to_msg()
        marker_line.type = marker_line.LINE_LIST
        marker_line.action = marker_line.ADD
        marker_line.scale.x = 0.1
        marker_line.scale.y = 0.1
        marker_line.scale.z = 0.1
        marker_line.color.a = 1.0
        marker_line.color.r = 0.0
        marker_line.color.g = 1.0
        marker_line.color.b = 0.0
        marker_line.lifetime = Duration(seconds=3, nanoseconds=5e2).to_msg()
        return marker_line

    def init_all_hand_msgs(self, _msg, count):
        _msg.hand_id = count
        _msg.palm = self.init_finger_points_msg()
        _msg.thumb_1 = self.init_finger_points_msg()
        _msg.thumb_2 = self.init_finger_points_msg()
        _msg.thumb_3 = self.init_finger_points_msg()
        _msg.thumb_4 = self.init_finger_points_msg()
        _msg.index_finger_1 = self.init_finger_points_msg()
        _msg.index_finger_2 = self.init_finger_points_msg()
        _msg.index_finger_3 = self.init_finger_points_msg()
        _msg.index_finger_4 = self.init_finger_points_msg()
        _msg.middle_finger_1 = self.init_finger_points_msg()
        _msg.middle_finger_2 = self.init_finger_points_msg()
        _msg.middle_finger_3 = self.init_finger_points_msg()
        _msg.middle_finger_4 = self.init_finger_points_msg()
        _msg.ring_finger_1 = self.init_finger_points_msg()
        _msg.ring_finger_2 = self.init_finger_points_msg()
        _msg.ring_finger_3 = self.init_finger_points_msg()
        _msg.ring_finger_4 = self.init_finger_points_msg()
        _msg.baby_finger_1 = self.init_finger_points_msg()
        _msg.baby_finger_2 = self.init_finger_points_msg()
        _msg.baby_finger_3 = self.init_finger_points_msg()
        _msg.baby_finger_4 = self.init_finger_points_msg()
        return _msg

    def add_point_to_marker(self, finger_points_msg):
        p = Point()
        p.x = float((finger_points_msg.x / self.width) * self.point_range)
        p.y = float((finger_points_msg.y / self.height) * self.point_range)
        p.z = 0.0
        return p

    def valid_marker_point(self, finger_points_msg):
        if math.isnan(finger_points_msg.x) or math.isnan(finger_points_msg.y):
            return False
        return True

    def parse_k(self):
        image_idx = 0
        try:
            count = int(self.counts[image_idx])
            primary_msg = HandPoseDetection()
            for i in range(count):
                primary_msg.hand_id = i
                primary_msg = self.init_all_hand_msgs(_msg=primary_msg, count=i)
                marker_joints = self.init_markers_spheres()
                marker_skeleton = self.init_markers_lines()
                for k in range(21):
                    _idx = self.objects[image_idx, i, k]
                    if _idx >= 0:
                        _location = self.peaks[image_idx, k, _idx, :]
                        if k == 0:
                            primary_msg.palm = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.palm))
                            self.get_logger().info(
                                "Finger Point Detected: Palm at X:{}, Y:{}".format(primary_msg.palm.x,
                                                                                primary_msg.palm.y))
                        if k == 1:
                            primary_msg.thumb_1 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.thumb_1))
                            self.get_logger().info(
                                "Finger Point Detected: Thumb:1 at X:{}, Y:{}".format(primary_msg.thumb_1.x,
                                                                                    primary_msg.thumb_1.y))
                        if k == 2:
                            primary_msg.thumb_2 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.thumb_2))
                            self.get_logger().info(
                                "Finger Point Detected: Thumb: 2 at X:{}, Y:{}".format(primary_msg.thumb_2.x,
                                                                                     primary_msg.thumb_2.y))
                            if self.valid_marker_point(primary_msg.thumb_1):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_1))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_2))

                        if k == 3:
                            primary_msg.thumb_3 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.thumb_3))
                            self.get_logger().info(
                                "Finger Point Detected: Thumb: 3 at X:{}, Y:{}".format(primary_msg.thumb_3.x,
                                                                                       primary_msg.thumb_3.y))
                            if self.valid_marker_point(primary_msg.thumb_2):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_2))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_3))

                        if k == 4:
                            primary_msg.thumb_4 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.thumb_4))
                            self.get_logger().info(
                                "Finger Point Detected: Thumb: 4 at X:{}, Y:{}".format(primary_msg.thumb_4.x,
                                                                                       primary_msg.thumb_4.y))
                            if self.valid_marker_point(primary_msg.thumb_3):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_3))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_4))

                            if self.valid_marker_point(primary_msg.palm):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.palm))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.thumb_4))

                        if k == 5:
                            primary_msg.index_finger_1 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.index_finger_1))
                            self.get_logger().info(
                                "Finger Point Detected: Index Finger:1 at X:{}, Y:{}".format(primary_msg.index_finger_1.x,
                                                                                      primary_msg.index_finger_1.y))

                        if k == 6:
                            primary_msg.index_finger_2 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.index_finger_2))
                            self.get_logger().info(
                                "Finger Point Detected: Index Finger:2  at X:{}, Y:{}".format(primary_msg.index_finger_2.x,
                                                                                       primary_msg.index_finger_2.y))
                            if self.valid_marker_point(primary_msg.index_finger_1):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_1))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_2))

                        if k == 7:
                            primary_msg.index_finger_3 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.index_finger_3))
                            self.get_logger().info(
                                "Finger Point Detected: Index Finger: 3 at X:{}, Y:{}".format(primary_msg.index_finger_3.x,
                                                                                       primary_msg.index_finger_3.y))
                            if self.valid_marker_point(primary_msg.index_finger_2):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_2))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_3))

                        if k == 8:
                            primary_msg.index_finger_4 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.index_finger_4))
                            self.get_logger().info(
                                "Finger Point Detected: Index Finger: 4 at X:{}, Y:{}".format(primary_msg.index_finger_4.x,
                                                                                       primary_msg.index_finger_4.y))
                            if self.valid_marker_point(primary_msg.index_finger_3):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_3))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_4))

                            if self.valid_marker_point(primary_msg.palm):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.palm))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.index_finger_4))

                        if k == 9:
                            primary_msg.middle_finger_1 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.middle_finger_1))
                            self.get_logger().info(
                                "Finger Point Detected: Middle Finger:1 at X:{}, Y:{}".format(
                                    primary_msg.middle_finger_1.x,
                                    primary_msg.middle_finger_1.y))

                        if k == 10:
                            primary_msg.middle_finger_2 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.middle_finger_2))
                            self.get_logger().info(
                                "Finger Point Detected: Middle Finger:2  at X:{}, Y:{}".format(
                                    primary_msg.middle_finger_2.x,
                                    primary_msg.middle_finger_2.y))
                            if self.valid_marker_point(primary_msg.middle_finger_1):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_1))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_2))

                        if k == 11:
                            primary_msg.middle_finger_3 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.middle_finger_3))
                            self.get_logger().info(
                                "Finger Point Detected: Middle Finger: 3 at X:{}, Y:{}".format(
                                    primary_msg.middle_finger_3.x,
                                    primary_msg.middle_finger_3.y))
                            if self.valid_marker_point(primary_msg.middle_finger_2):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_2))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_3))

                        if k == 12:
                            primary_msg.middle_finger_4 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.middle_finger_4))
                            self.get_logger().info(
                                "Finger Point Detected: Middle Finger: 4 at X:{}, Y:{}".format(
                                    primary_msg.middle_finger_4.x,
                                    primary_msg.middle_finger_4.y))
                            if self.valid_marker_point(primary_msg.middle_finger_3):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_3))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_4))

                            if self.valid_marker_point(primary_msg.palm):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.palm))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.middle_finger_4))

                        if k == 13:
                            primary_msg.ring_finger_1 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.ring_finger_1))
                            self.get_logger().info(
                                "Finger Point Detected: Ring Finger:1 at X:{}, Y:{}".format(primary_msg.ring_finger_1.x,
                                                                                            primary_msg.ring_finger_1.y))

                        if k == 14:
                            primary_msg.ring_finger_2 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.ring_finger_2))
                            self.get_logger().info(
                                "Finger Point Detected: Ring Finger:2  at X:{}, Y:{}".format(
                                    primary_msg.ring_finger_2.x,
                                    primary_msg.ring_finger_2.y))
                            if self.valid_marker_point(primary_msg.ring_finger_1):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_1))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_2))

                        if k == 15:
                            primary_msg.ring_finger_3 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.ring_finger_3))
                            self.get_logger().info(
                                "Finger Point Detected: Ring Finger: 3 at X:{}, Y:{}".format(
                                    primary_msg.ring_finger_3.x,
                                    primary_msg.ring_finger_3.y))
                            if self.valid_marker_point(primary_msg.ring_finger_2):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_2))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_3))

                        if k == 16:
                            primary_msg.ring_finger_4 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.ring_finger_4))
                            self.get_logger().info(
                                "Finger Point Detected: Ring Finger: 4 at X:{}, Y:{}".format(
                                    primary_msg.ring_finger_4.x,
                                    primary_msg.ring_finger_4.y))
                            if self.valid_marker_point(primary_msg.ring_finger_3):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_3))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_4))

                            if self.valid_marker_point(primary_msg.palm):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.palm))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.ring_finger_4))

                        if k == 17:
                            primary_msg.baby_finger_1 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.baby_finger_1))
                            self.get_logger().info(
                                "Finger Point Detected: Baby Finger:1 at X:{}, Y:{}".format(primary_msg.baby_finger_1.x,
                                                                                            primary_msg.baby_finger_1.y))

                        if k == 18:
                            primary_msg.baby_finger_2 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.baby_finger_2))
                            self.get_logger().info(
                                "Finger Point Detected: Baby Finger:2  at X:{}, Y:{}".format(
                                    primary_msg.baby_finger_2.x,
                                    primary_msg.baby_finger_2.y))
                            if self.valid_marker_point(primary_msg.baby_finger_1):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_1))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_2))

                        if k == 19:
                            primary_msg.baby_finger_3 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.baby_finger_3))
                            self.get_logger().info(
                                "Finger Point Detected: Baby Finger: 3 at X:{}, Y:{}".format(
                                    primary_msg.baby_finger_3.x,
                                    primary_msg.baby_finger_3.y))
                            if self.valid_marker_point(primary_msg.baby_finger_2):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_2))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_3))

                        if k == 20:
                            primary_msg.baby_finger_4 = self.write_finger_points_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.baby_finger_4))
                            self.get_logger().info(
                                "Finger Point Detected: Baby Finger: 4 at X:{}, Y:{}".format(
                                    primary_msg.baby_finger_4.x,
                                    primary_msg.baby_finger_4.y))
                            if self.valid_marker_point(primary_msg.baby_finger_3):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_3))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_4))

                            if self.valid_marker_point(primary_msg.palm):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.palm))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.baby_finger_4))

                        self.publish_pose.publish(primary_msg)
                        self.hand_skeleton_pub.publish(marker_skeleton)
                        self.hand_joints_pub.publish(marker_joints)

        except:
            pass
