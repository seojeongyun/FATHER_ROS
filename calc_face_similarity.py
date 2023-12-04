#!/usr/bin/env python3

import sys

sys.path.append("/home/kcmee/PycharmProjects/")
sys.path.append("/home/kcmee/PycharmProjects/groupface_hr/")

from demo_core import load_gallery_probe
from demo_core import face_similarity

import cv2
import rospy
import torch.nn.functional as F
import torch
import numpy as np

import os

import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from function import imgmsg_to_cv2, cv2_to_imgmsg

class calc_face_sim:
    def __init__(self):
        self.cropped_face_sub = rospy.Subscriber('/face_detector/cropped_face_for_similarity', Image, self.callback)
        self.update_face_vec_flag_sub = rospy.Subscriber("/detected_face_view/update_face_vec_flag", Int32, self.update_callback)
        #
        self.result_image_pub = rospy.Publisher("/calc_face_similarity/result_image", Image, queue_size = 10)
        #
        self.image_size = 224
        self.device = torch.device('cuda:0')
        self.MAX_NUM_TOP_IDX = 5
        #
        g_files = self.get_gallery_path(
                dict(path="/storage/hrlee/vggface2/demo/gallery", max_num_imgs=10, max_num_gallery_imgs=10)
                )
        #
        network_config = dict(
                # Parameters for resnet
        	resnet=18,
        	# Parameters for capsule network
        	capdimen=48,
        	numpricap=512,
        	predcapdimen=64,
        	num_final_cap=64,
        	# Parameters for GroupFace Structure
        	feature_dim=512, groups=5,
        	# Other Parameters
        	training_gpu='4080',
        	device=self.device,
        	)
        network_config.setdefault('weight', 
        	"/home/kcmee/PycharmProjects/groupface_hr/checkpoints/best_res18caps_group5_featdim1024_top1_0_894_harmonic_0_924.pth"
        )
        #
        self.face_similarity = self.get_model(network_config, g_files)
        
    
    def get_gallery_path(self, gallery):
    	# LOAD GALLERY FILES & COMPUTE FACE VECTORS
        g_files, _ = load_gallery_probe(gallery, "/storage/hrlee/vggface2/demo/probe", split=False)
        return g_files
    	
    	
    def get_model(self, network_config, gallery_files):
        # LOAD FACE SIMILARITY NETWORK
        FACE_SIMILARITY = face_similarity(
                dict(
                    imgsize=self.image_size,
                    network_cfg=network_config
                    )
                )
        # MAKE GALLERY FEATURES' DATABASE
        FACE_SIMILARITY.make_gallery_feats_database(gallery_files)
        return FACE_SIMILARITY

        
        
    def callback(self, data):
        try:
            data.encoding = 'bgr8'
            cropped_face = torch.tensor(imgmsg_to_cv2(data).transpose(2, 0, 1))
            top5_ids, top5_scores, resized_probe_img, top5_g_imgs = (
            	self.face_similarity.find_top5_face_ids(cropped_face, self.MAX_NUM_TOP_IDX, verbose=False))
            #
            print("top5_scores: ", top5_scores)
            result_img = self.face_similarity.view_result(resized_probe_img, None, top5_ids, top5_scores, top5_g_imgs)
            result_msg = cv2_to_imgmsg(result_img)
            self.result_image_pub.publish(result_msg)
            #
            plt.figure()
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.show()
            
        except:
            print("ERROR : CALCULATE_FACE_SIMILARITY CALLBACK\n")
            
    
    def update_callback(self, data):
        try:
            if data.data == 1:
                print("START UPDATE")
                g_files = self.get_gallery_path(dict(path="/storage/hrlee/vggface2/demo/gallery", max_num_imgs=10, max_num_gallery_imgs=10))
                print("START UPDATE")
                self.face_similarity.make_gallery_feats_database(g_files)
        except:
            print("update error")
                
                
    @staticmethod
    def pad_to_square(image, pad_value=0):
        _, h, w = image.shape

        difference = abs(h - w)

        # (top, bottom) padding or (left, right) padding
        if h <= w:
            top = difference // 2
            bottom = difference - difference // 2
            pad = [0, 0, top, bottom]
        else:
            left = difference // 2
            right = difference - difference // 2
            pad = [left, right, 0, 0]


        # Add padding
        image = F.pad(torch.tensor(image), pad, mode='constant', value=pad_value)
        return image, pad
        
    @staticmethod
    def resize(image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)
        

def main():
    print('\033[96m' + '[START] ROS NODE: CALC_FACE_SIM' + '\033[0m')
    img_node = calc_face_sim()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: CALC_FACE_SIM" + '\033[0m')
    rospy.init_node('img_node', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
