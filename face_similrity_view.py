#!/usr/bin/env python3

import cv2
import rospy

from function import imgmsg_to_cv2, cv2_to_imgmsg
from sensor_msgs.msg import Image

class face_similarity_view:
    def __init__(self):
        self.result_image_sub = rospy.Subscriber("/calc_face_similarity/result_image", Image, self.callback)

    def callback(self, data):
        try:
#            data.encoding = 'bgr8'
            result_image = imgmsg_to_cv2(data)
            cv2.imshow('result', result_image)
            cv2.waitKey()  
        except:
            print("ERROR : FACE_SIMILARITY_VIEW CALLBACK\n")
 
def main():
    print('\033[96m' + '[START] ROS NODE: CALC_FACE_RESULT_VIEW' + '\033[0m')
    img_node = face_similarity_view()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: CALC_FACE_RESULT_VIEW" + '\033[0m')
    rospy.init_node('img_node', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

