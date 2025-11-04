#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict

import rospy
import cv2
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results

from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
import tf
import numpy as np

class Yolov8Node(object):
    def __init__(self) -> None:
        # 노드 초기화
        rospy.init_node("yolov8_node")

        # params (private namespace 사용 권장: ~param_name)
        self.model_path = rospy.get_param("~model", "yolov8m.pt")
        self.device = rospy.get_param("~device", "cuda:0")
        self.threshold = rospy.get_param("~threshold", 0.4)
        self.enable = rospy.get_param("~enable", True)

        # topics
        self.image_topic = rospy.get_param("~image_topic", "/go2_front_camera/image_raw")
        self.stage_topic = rospy.get_param("~stage_topic", "/stage")
        self.image_out_topic = rospy.get_param("~image_out_topic", "/yolov8/image_out")
        self.traffic_state_topic = rospy.get_param("~traffic_state_topic", "/traffic_state")
        self.traffic_command_topic = rospy.get_param("~traffic_command_topic", "/traffic_command")

        # utils
        self.bridge = CvBridge()
        self.yolo = YOLO(self.model_path)
        try:
            # 일부 모델에서만 fuse 지원. 실패해도 무시 가능
            self.yolo.fuse()
        except Exception as e:
            rospy.logwarn(f"YOLO fuse skipped: {e}")

        # pubs
        self.pub_image = rospy.Publisher(self.image_out_topic, Image, queue_size=1)
        self.pub_traffic = rospy.Publisher(self.traffic_state_topic, String, queue_size=1)
        self.pub_traffic_command = rospy.Publisher(self.traffic_command_topic, Bool, queue_size=1)

        # subs
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.sub_odom = rospy.Subscriber("liorf_localization/mapping/odometry_base",
            Odometry,
            self.odometry_callback,
            queue_size=1
        )

        key_region_dicts = rospy.get_param("/key_region")
        for key_region_dict in key_region_dicts:
            key_name = key_region_dict["name"]
            if key_name == "traffic":
                x = key_region_dict["x"]
                y = key_region_dict["y"]
                yaw = key_region_dict["yaw"]
                short_range = key_region_dict["short_axis"]
                print("received rosparam, x,y: ", x, y)

        self.activate_region_yaw = float(yaw)
        self.activate_region_xy = np.array([x,y])
        self.activate_region_radii = float(short_range)

        rospy.loginfo("YOLOv8 ROS1 node started")
        self.enable = False
        self.traffic_command_counter = 0
        self.traffic_command = True

        # yolo node status
        self.odom_received = False

        # 
        cv_image = np.ones([480,640,3])
        results_list = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device
        )
        print("ready")
        print("waiting for image")

    def odometry_callback(self, msg):
        self.current_odometry = msg

        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        # heading = msg.pose.pose.orientation

        current_pose = np.array([pos_x,pos_y])
        distance = np.linalg.norm(current_pose - self.activate_region_xy)
        
        q = msg.pose.pose.orientation
        quaternion = (q.x, q.y, q.z, q.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

        # yaw 차이 (라디안)
        yaw_diff = yaw - self.activate_region_yaw

        # print(distance, yaw_diff)

        self.odom_received = True

        if distance < self.activate_region_radii:
            # print("yolo on")
            self.enable = True
        else:
            # print("yolo off")
            self.enable = False

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return hypothesis_list

        # 안전하게 index 기반 접근
        n = len(boxes)
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            hypothesis = {
                "class_id": cls_id,
                "class_name": self.yolo.names[cls_id],
                "score": score
            }
            hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results):
        boxes_list = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return boxes_list

        n = len(boxes)
        for i in range(n):
            box_xywh = boxes.xywh[i]  # tensor [x, y, w, h]
            boxes_list.append(box_xywh)
        return boxes_list

    def image_cb(self, msg: Image):
        """
        - 입력: sensor_msgs/Image (BGR)
        - 출력: 시각화 이미지 퍼블리시, 신호등 상태 퍼블리시
        """
        light_state = 2  # 0: red, 1: green, 2: unknown

        try:
            if self.enable:
                # ROS Image -> OpenCV (BGR)
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                cv_image = cv2.resize(cv_image, (960,540))
                # print("run yolo")
                # YOLO 추론 (Ultralytics는 BGR/RGB 모두 처리 가능)
                results_list = self.yolo.predict(
                    source=cv_image,
                    verbose=False,
                    stream=False,
                    conf=self.threshold,
                    device=self.device
                )
                results: Results = results_list[0].cpu()

                # (옵션) 결과 파싱
                _ = self.parse_hypothesis(results)
                _ = self.parse_boxes(results)

                # 시각화용 이미지 (BGR 유지)
                cv_image_vis = cv_image.copy()

                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    n = len(boxes)
                    for i in range(n):
                        xyxy = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                        class_id = int(boxes.cls[i].item())
                        score = float(boxes.conf[i].item())
                        label = f"{self.yolo.names[class_id]} {score:.2f}"

                        x1, y1, x2, y2 = list(map(int, xyxy))
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(cv_image_vis.shape[1]-1, x2)
                        y2 = min(cv_image_vis.shape[0]-1, y2)

                        # traffic light 클래스일 때 crop하여 상태 판별
                        if self.yolo.names[class_id] == "traffic light":
                            crop = cv_image_vis[y1:y2, x1:x2]  # BGR crop
                            if crop.size > 0:
                                light_state = self.green_or_red(crop)

                        # 박스/라벨 그리기 (BGR)
                        cv2.rectangle(cv_image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image_vis, label, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 퍼블리시 (bgr8)
                result_msg = self.bridge.cv2_to_imgmsg(cv_image_vis, encoding="bgr8")
                result_msg.header = msg.header  # 타임스탬프/프레임 유지
                self.pub_image.publish(result_msg)

            # 신호등 상태 퍼블리시 (enable 여부와 무관하게 매 프레임 보냄: 원 코드와 동일)
            traffic_state_msg = String()
            if light_state == 0:
                # print("traffic light: red")
                traffic_state_msg.data = "red"
                self.traffic_command_counter = 0
                self.traffic_command = False
                print("red")
            elif light_state == 1:
                # rospy.logdebug("traffic light: green")
                traffic_state_msg.data = "green"
                self.traffic_command_counter += 1
                print("green")
            else:
                # rospy.logdebug("traffic light: unknown")
                traffic_state_msg.data = "unknown"
                if self.enable:
                    print("yolo on, cannot catch traffic light")
                # print("unknown or yolo off")
            self.pub_traffic.publish(traffic_state_msg)
            
            # print out yolo node status
            print(f"YOLO node status (odom_received/enable/traffic)\n\
                   {self.odom_received}/{self.enable}/{traffic_state_msg.data}")
 
            if (self.traffic_command_counter > 3):
                self.traffic_command = True
                
            if (self.traffic_command):
                traffic_command_msg = Bool()
                traffic_command_msg.data = True
                self.pub_traffic_command.publish(traffic_command_msg)
            else:
                traffic_command_msg = Bool()
                traffic_command_msg.data = False
                self.pub_traffic_command.publish(traffic_command_msg)

        except Exception as e:
            rospy.logerr(f"image_cb error: {e}")

    def green_or_red(self, img):
        """
        Input: cropped region (BGR)
        Return: 0(red) / 1(green) / 2(unknown)
        """
        # BGR -> HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # red (두 구간)
        lower_red1 = (0, 100, 100)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 100, 100)
        upper_red2 = (180, 255, 255)

        # green
        lower_green = (40, 100, 100)
        upper_green = (90, 255, 255)

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)

        if red_pixels > green_pixels and red_pixels > 50:
            return 0
        elif green_pixels > red_pixels and green_pixels > 50:
            return 1
        else:
            return 2


def main():
    node = Yolov8Node()
    # rospy.Rate를 쓰지 않고 spin으로 콜백만 처리
    rospy.spin()


if __name__ == "__main__":
    main()
