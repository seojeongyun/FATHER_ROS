#!/usr/bin/env python3

import cv2
import sys

import rospy
from sensor_msgs.msg import Image
from ring_buffer import CircularBuffer

class fps_controller:
    def __init__(self):
        self.Drone = False
        self.color_sub = rospy.Subscriber("/drone/image_raw", Image, self.callback) if self.Drone else rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        # self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        self.result_pub = rospy.Publisher('/fps_controller/image_raw', Image, queue_size=1)
        #
        
        self.circular_buff = CircularBuffer(50)
        self.count = 0

    def callback(self, data):
        try:
            if self.Drone:
                self.result_pub.publish(data)
                
            else:
                self.count += 1
                if self.count % 1 == 0:
                    self.result_pub.publish(data)
                elif self.count >= 100:
                    self.count = 0                
        except:
            print("ERROR : FPS_CONTROLLER CALLBACK\n")


def main():
    print('\033[96m' + '[START] ROS NODE: FPS_CONTROLLER' + '\033[0m')
    img_node = fps_controller()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: FPS_CONTROLLER" + '\033[0m')
    rospy.init_node('fps_controller', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

