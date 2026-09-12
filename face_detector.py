#!/usr/bin/env python3

import cv2
import sys
sys.path.append("/home/kcmee/PycharmProjects/")
sys.path.append("/home/kcmee/PycharmProjects/YOLOv6/")

import time
import torch
import rospy
import matplotlib.pyplot as plt
import numpy as np
import keyboard 

from function import imgmsg_to_cv2, cv2_to_imgmsg
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from YOLOv6.yolov6.core.demo import demo


# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 10
RESIZE_SIZE = 320

ONNX = False
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/home/parksungjun/Downloads/ours_SW_150.onnx"
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/home/kcmee/Downloads/6n_960.pt"
CONF_THRES = 0.35
IOU_THRES = 0.5
NMS_DET = 100
GPU_ID = '0'
# DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
DEVICE = torch.device('cuda:0')
VIEW = True
NET_TYPE='6n'

FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET, NET_TYPE)
    

class cam_image_subscribe_publish:
    def __init__(self):
        self.color_sub = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback) # from astra camera
        #
        self.result_pub = rospy.Publisher('/face_detector/image_result', Image, queue_size=10) # publish not cropped image with bounding box to detected_face_view.py
        #
        self.cropped_img_pub_for_sim = rospy.Publisher('/face_detector/cropped_face_for_similarity', Image, queue_size=10) # publish cropped image to calc_face_similarity.py when press the key g
        self.cropped_img_pub_for_gallery = rospy.Publisher('/face_detector/cropped_face_for_update_gallery', Image, queue_size=10) # publish cropped image to detected_face_view.py when press the key n
        #
        self.save_flag_on_pub = rospy.Publisher('/face_detector/save_flag_on', Int32, queue_size=10) # publish save flag to detected_face_view.py when press the key s
        self.save_flag_off_pub = rospy.Publisher('/face_detector/save_flag_off', Int32, queue_size=10) # publish save flag to detected_face_view.py when press the key e
            
    def face_crop(self, video_image, out_boxes):
        cropped_face_list = [] 
        for idx in range(len(out_boxes[0])):
            box = out_boxes[0][idx][:4].cpu().numpy()
            box = box.round().astype(np.int32).tolist()
            img_w, img_h = video_image[0].shape[:2]
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            img = video_image[0][y1: y2, x1: x2]
            cropped_face_list.append(img)
        return cropped_face_list
    
    def cropped_face_publisher(self, cropped_images):
        if len(cropped_images) == 1:
            face_image = cropped_images[0]
            face_img_msg = cv2_to_imgmsg(face_image)
            self.cropped_img_pub.publish(face_img_msg)
            
        else:
            for idx in range(len(cropped_images)):
                face_image = cropped_images[idx]
                face_img_msg = cv2_to_imgmsg(face_image)
                self.cropped_img_pub.publish(face_img_msg)

    def callback(self, data):
        try:
            flag = 0
            keyboard.init()
            data.encoding = 'bgr8'
            cv_image = imgmsg_to_cv2(data)
            cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
            video_img = [cv_image]
            bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(video_img)
           
            img_w_box, cropped_images = FACE_DETECTOR.post_processing(bboxes, video_img)
            detected_face = img_w_box[0]
            detected_face = cv2.resize(detected_face, (900,900))
            
            img_msg = cv2_to_imgmsg(detected_face)
            self.result_pub.publish(img_msg)
            
            if keyboard.getKey('0'):
                time.sleep(0.5)
                if len(cropped_images) > 0:
                    for i in range(len(cropped_images)):
                        cropped_face = cropped_images[i]
                        img_msg_cropped = cv2_to_imgmsg(cropped_face)
                        self.cropped_img_pub_for_sim.publish(img_msg_cropped)
            
            if keyboard.getKey('1'):
                self.save_flag_on_pub.publish(1)
                
            if keyboard.getKey('2'):
                time.sleep(0.05)
                if len(cropped_images) > 0:
                    cropped_face = cropped_images[0]
                    img_msg_cropped = cv2_to_imgmsg(cropped_face)
                self.cropped_img_pub_for_gallery.publish(img_msg_cropped)

            if keyboard.getKey('3'):
                time.sleep(0.5)
                self.save_flag_off_pub.publish(1)
                
        except:
            print("ERROR : FACE_DETECTOR CALLBACK\n")
           

def main():
    print('\033[96m' + '[START] ROS NODE: CAM_IMAGE_SUBSCRIBER' + '\033[0m')
    img_node = cam_image_subscribe_publish()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: CAM_IMAGE_SUBSCRIBER" + '\033[0m')
    rospy.init_node('img_node', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

