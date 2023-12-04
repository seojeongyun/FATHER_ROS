#!/usr/bin/env python3

import cv2
import rospy
import keyboard

from djitellopy import Tello
from sensor_msgs.msg import Image
from function import imgmsg_to_cv2, cv2_to_imgmsg
from ring_buffer import CircularBuffer

class drone:
    def __init__(self):
        self.use_video = True
        self.use_keyboard = False
        #
        self.buffer = CircularBuffer(50)
        self.drone_img_pub = rospy.Publisher('/drone/image_raw', Image, queue_size=10)
        #
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        #
        print(self.drone.get_battery())
        
    def get_frame(self):
        try:
            if self.use_video:
                raw_image = self.drone.get_frame_read().frame
                raw_image = cv2.resize(raw_image, (480,640))         
                self.buffer.enqueue(raw_image)               
                print(raw_image.shape)           
                return self.buffer.dequeue()
	    
        except:
            print("ERROR : GET_FRAME\n")
                
    #def get_key(self):
        # if self.use_keyboard:
 

if __name__=='__main__':
    print('\033[96m' + '[START] ROS NODE: DRONE' + '\033[0m')
    tello = drone()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: DRONE" + '\033[0m')
    rospy.init_node('drone', anonymous=True)
        

    while not rospy.is_shutdown():
        raw_image = tello.get_frame()
        raw_image_msg = cv2_to_imgmsg(raw_image)	 
        #
        tello.drone_img_pub.publish(raw_image_msg) 


        
        
        
