# Make ros workspace
mkdir -p ~/ros_ws/src && cd ~/ros_ws/src 
catkin_init_workspace
cd ~/ros_ws
catkin_make
source ./devel/setup.bash



# Make package
cd ~/ros_ws/src
catkin_create_pkg video_stream rospy std_msgs roscpp




# run FATHER 
0. cd ~/ros_ws && source ./devel/setup.bash                  #  in all terminal
1. roscore                                                   #  terminal 1
2. roslaunch astra_camera astra.pro.plus.launch              #  terminal 2
   # refer : https://github.com/orbbec/ros_astra_camera
3. rosrun video_stream fps_controller.py                     #  terminal 3
4. rosrun video_stream face_detector.py                      #  terminal 4
5. rosrun video_stream detected_face_view.py                 #  terminal 5
6. rosrun video_stream calc_face_similarity.py               #  terminal 6

!!  There are all the python scripts (3~6) in ~/ros_ws/src/video_script/src
