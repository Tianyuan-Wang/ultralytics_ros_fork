#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

class MultiTrackerNode:
    def __init__(self):
        """

        """
        ''' Get ROS parameters '''
        self.yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.left_input_topic = rospy.get_param("~left_input_topic", "/left/image_raw")
        self.right_input_topic = rospy.get_param("~right_input_topic", "/right/image_raw")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)

        ''' Topics for output data '''
        self.left_result_topic = rospy.get_param("~left_result_topic", "/left/yolo_result")
        self.left_result_image_topic = rospy.get_param("~left_result_image_topic", "/left/yolo_image")
        self.right_result_topic = rospy.get_param("~right_result_topic", "/right/yolo_result")
        self.right_result_image_topic = rospy.get_param("~right_result_image_topic", "/right/yolo_image")

        ''' Instantiate YOLO model, which is shared to left and right channels '''
        rospy.loginfo("Loading YOLO model: %s", self.yolo_model)
        self.model = YOLO(self.yolo_model)
        self.model.fuse()

        ''' Subscribe to input image topics '''
        self.bridge = cv_bridge.CvBridge()
        self.left_sub = rospy.Subscriber(self.left_input_topic, Image, self.left_image_callback, queue_size=1, buff_size=2**24)
        self.right_sub = rospy.Subscriber(self.right_input_topic, Image, self.right_image_callback, queue_size=1, buff_size=2**24)

        ''' Publish results '''
        self.left_result_pub = rospy.Publisher(self.left_result_topic, YoloResult, queue_size=1)
        self.left_result_image_pub = rospy.Publisher(self.left_result_image_topic, Image, queue_size=1)
        self.right_result_pub = rospy.Publisher(self.right_result_topic, YoloResult, queue_size=1)
        self.right_result_image_pub = rospy.Publisher(self.right_result_image_topic, Image, queue_size=1)

        rospy.loginfo("MultiTrackerNode init done.")

    def left_image_callback(self, msg):
        """
        Callback function to process left channel image
        :param msg:
        :return:
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        self.publish_result(msg.header, results, self.left_result_pub, self.left_result_image_pub)

    def right_image_callback(self, msg):
        """
        Callback function to process right channel image
        :param msg:
        :return:
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        self.publish_result(msg.header, results, self.right_result_pub, self.right_result_image_pub)

    def publish_result(self, header, results, pub_result, pub_image):
        """
        A utility function to publish results.
        :param header:
        :param results:
        :param pub_result:
        :param pub_image:
        :return:
        """
        if results is not None and len(results) > 0:
            yolo_result_msg = YoloResult()
            yolo_result_msg.header = header
            # Detection
            detections = Detection2DArray()
            bounding_box = results[0].boxes.xywh
            classes = results[0].boxes.cls
            confidence_score = results[0].boxes.conf

            for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                detection = Detection2D()
                detection.bbox.center.x = float(bbox[0])
                detection.bbox.center.y = float(bbox[1])
                detection.bbox.size_x   = float(bbox[2])
                detection.bbox.size_y   = float(bbox[3])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(cls)
                hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detections.detections.append(detection)

            yolo_result_msg.detections = detections

            ''' Draw an overlay for visualisation '''
            plotted_image = results[0].plot()
            result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")

            ''' Publish them '''
            pub_result.publish(yolo_result_msg)
            pub_image.publish(result_image_msg)

def main():
    rospy.init_node("multi_tracker_node")
    node = MultiTrackerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
