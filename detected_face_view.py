#!/usr/bin/env python3

import os
import cv2
import sys
import rospy
import glob
import numpy as np

from function import imgmsg_to_cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from ring_buffer import CircularBuffer
from function import imgmsg_to_cv2, cv2_to_imgmsg

import sys

class detected_face_subscriber:
    def __init__(self):
        self.color_sub = rospy.Subscriber("/face_detector/image_result", Image, self.callback) # Subscribe not cropped image with bounding box from face_detector
        #
        self.cropped_face_sub = rospy.Subscriber('/face_detector/cropped_face_for_update_gallery', Image, self.save_callback) # Subscribe cropped image from face_detector
        self.save_flag_on_sub = rospy.Subscriber("/face_detector/save_flag_on", Int32, self.save_flag_on_callback) # Subscribe either save_flag_on or save_flag_off 
        self.save_flag_off_sub = rospy.Subscriber("/face_detector/save_flag_off", Int32, self.save_flag_off_callback) # Subscribe either save_flag_on or save_flag_off 
        #
        self.update_face_vec_flag = rospy.Publisher("/detected_face_view/update_face_vec_flag", Int32, queue_size=10)
        #
        self.buffer = CircularBuffer(50)
        #
        self.save_flag = 0
        self.counter = 0
        #
    def callback(self, data):
        try:
            data.encoding = 'bgr8'
            self.cv_image = imgmsg_to_cv2(data)
            self.buffer.enqueue(self.cv_image)
            cv2.imshow('FACE_DETECT', self.buffer.dequeue())
            cv2.waitKey(int(1000/30))
           
        except:
            print("DETECTED FACE IMAGE CALLBACK ERROR\n")
            
    def save_flag_on_callback(self, msg): # press key 0
        try:
            if self.save_flag == 0 and msg.data == 1:
                self.save_flag = 1
                #
                self.name = input("What is your name ? :")
                #self.name = sys.stdin.readline
                #print("What is your name? : ", self.name())
                self.save_path = "/storage/hrlee/vggface2/demo/gallery" + '/' + str(self.name)
                #
                if not os.path.exists(self.save_path):
                    print(self.save_path)
                    os.makedirs(self.save_path, exist_ok=True)
                #
                self.counter = len(glob.glob(self.save_path + '/*'))

            else: pass
             
        except:
            print("Not set self.save_flag to 1")

    def save_flag_off_callback(self, msg): # press key 3
        try:
            continue_ = input("Add more galleries ? (y/n)")
            if self.save_flag == 1 and msg.data == 1 and continue_ == 'n':
                self.save_flag = 0
                #
                self.name = None
                self.save_path = None
                self.counter = -1
                #
                print("self.save_path : {}".format(self.save_path))
                print("self.name : {}".format(self.name))
                print("self.counter : {}".format(self.counter))
                self.update_face_vec_flag.publish(1)
            elif self.save_flag == 1 and msg.data == 1 and continue_ == 'y':
                self.save_flag = 0
                pass
                
                
        except:
            print("Not set self.save_flag to 0")
        
    def save_callback(self, data): # # press key 2
        try:
            print(self.save_flag==1)
            if self.save_flag == 1:
                cropped_face = imgmsg_to_cv2(data)
                print(" CROPPED FACE IS SAVING.. ")
                cv2.imwrite(self.save_path + '/' + self.name + '_{}.jpg'.format(self.counter), cropped_face)
                self.counter += 1
                
        except:
            print("ERROR : DETECTED_FACE_SUBSCRIBER SAVE_CALLBACK\n")
                 
               
def main():
    print('\033[96m' + '[START] ROS NODE: FACE IMAGE SUBSCRIBER' + '\033[0m')
    img_node = detected_face_subscriber()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: FACE IMAGE SUBSCRIBER" + '\033[0m')
    rospy.init_node('detected_face', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
