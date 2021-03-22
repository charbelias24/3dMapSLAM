#!/bin/bash
declare -a bag_files=("MH_03_medium.bag" "MH_05_difficult.bag" "V2_03_difficult.bag")

echo "Publishing ${bag_files[$1]}"

rosbag play --pause ~/Documents/Datasets/EuRoC/${bag_files[$1]}  /cam0/image_raw:=/camera/left/image_raw /cam1/image_raw:=/camera/right/image_raw /imu0:=/imu

