rosrun --prefix 'gdb -ex run --args'  ORB_SLAM3 Stereo_Inertial Vocabulary/ORBvoc.txt ../../ZED/params/stereo-imu/zed2.yaml false /camera/left/image_raw:=/zed2/zed_node/left/image_rect_gray /imu:=/zed2/zed_node/imu/data /camera/right/image_raw:=/zed2/zed_node/right/image_rect_gray

