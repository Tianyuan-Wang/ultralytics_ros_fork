#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv_bridge
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

class MultiTrackerSynced:
    def __init__(self):
        """

        """
        ''' Get ROS parameters '''
        self.yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres  = rospy.get_param("~iou_thres", 0.45)
        self.max_det    = rospy.get_param("~max_det", 300)
        self.classes    = rospy.get_param("~classes", None)
        self.tracker    = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device     = rospy.get_param("~device", None)

        ''' Topics for input image streams '''
        self.left_input_topic  = rospy.get_param("~left_input_topic",  "/left/image_raw")
        self.right_input_topic = rospy.get_param("~right_input_topic", "/right/image_raw")

        ''' Topics for output data '''
        self.left_result_topic        = rospy.get_param("~left_result_topic",        "/left/yolo_result")
        self.left_result_image_topic  = rospy.get_param("~left_result_image_topic",  "/left/yolo_image")
        self.right_result_topic       = rospy.get_param("~right_result_topic",       "/right/yolo_result")
        self.right_result_image_topic = rospy.get_param("~right_result_image_topic", "/right/yolo_image")

        ''' Result plotting configuration parameters '''
        self.result_conf       = rospy.get_param("~result_conf",       True)
        self.result_line_width = rospy.get_param("~result_line_width", 1)
        self.result_font_size  = rospy.get_param("~result_font_size",  1)
        self.result_font       = rospy.get_param("~result_font",       "Arial.ttf")
        self.result_labels     = rospy.get_param("~result_labels",     True)
        self.result_boxes      = rospy.get_param("~result_boxes",      True)

        ''' Instantiate YOLO model, which is shared to left and right channels '''
        rospy.loginfo("Loading YOLO model: %s", self.yolo_model)
        self.model = YOLO(self.yolo_model)
        self.model.fuse()

        ''' Subscribe to input image topics and set up synchroniser'''
        self.bridge = cv_bridge.CvBridge()
        left_sub    = message_filters.Subscriber(self.left_input_topic,  Image)
        right_sub   = message_filters.Subscriber(self.right_input_topic, Image)
        # Create an approximate synchroniser
        sync = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub],
                                                           queue_size=10,
                                                           slop=0.1,
                                                           allow_headerless=True)
        sync.registerCallback(self.synced_image_callback)

        ''' Publish results '''
        self.left_result_pub  = rospy.Publisher(self.left_result_topic,         YoloResult, queue_size=1)
        self.left_image_pub   = rospy.Publisher(self.left_result_image_topic,   Image,      queue_size=1)
        self.right_result_pub = rospy.Publisher(self.right_result_topic,        YoloResult, queue_size=1)
        self.right_image_pub  = rospy.Publisher(self.right_result_image_topic,  Image,      queue_size=1)

        rospy.loginfo("MultiTrackerSynced init done.")

    def synced_image_callback(self, left_msg, right_msg):
        """
        Callback function to receive both left and right images and process them.
        left_msg.header.stamp and right_msg.header.stamp need to be within the threshold.
        """
        ''' Convert to cv2 image format for later processing '''
        left_cv_image  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding="bgr8")
        right_cv_image = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")

        ''' YOLO processing on left and right images '''
        left_results  = self.model.track(
            source=left_cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )

        right_results = self.model.track(
            source=right_cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )

        ''' Publish left and right results '''
        # 注意：如果要分开管理左右跟踪器，需要在 YOLO v8 中指定 tracker 重置策略
        # 这里简单地分别把图像扔给同一个 model.track()，内部会维护一个Tracker实例
        # 可能要在 model.track() 中加上 "persist=True" 之类，保证跟踪状态独立
        self.publish_result(left_msg.header,  left_results,  self.left_result_pub,  self.left_image_pub)
        self.publish_result(right_msg.header, right_results, self.right_result_pub, self.right_image_pub)

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
            plotted_image = results[0].plot(
                conf=self.result_conf,
                line_width=self.result_line_width,
                font_size=self.result_font_size,
                font=self.result_font,
                labels=self.result_labels,
                boxes=self.result_boxes,
            )
            result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")

            ''' Publish them '''
            pub_result.publish(yolo_result_msg)
            pub_image.publish(result_image_msg)

def main():
    rospy.init_node("multi_tracker_node")
    node = MultiTrackerSynced()
    rospy.spin()

if __name__ == "__main__":
    main()
