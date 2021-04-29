#!/bin/bash

if [[ "$1" == "zed" ]] 
then
	echo "Launching SLAM with ZED HD camera"
	roslaunch slam_rgbd_zed.launch zed_params:=/stereo/zed.yaml
elif [[ "$1" == "zed-vga" ]] 
then
	echo "Launching SLAM with ZED VGA camera"
	roslaunch slam_rgbd_zed.launch zed_params:=/stereo/zed-VGA.yaml
elif [[ "$1" == "zed2" ]] 
then
	echo "Launching SLAM with ZED2 HD camera"
	roslaunch slam_rgbd_zed.launch zed_params:=/stereo/zed2.yaml
elif [[ "$1" == "zedm" ]] 
then
	echo "Launching SLAM with ZEDm HD camera"
	roslaunch slam_rgbd_zed.launch zed_params:=/stereo/zedm.yaml
else
	echo "You must add one of these arguments:"
	echo "- zed"
	echo "- zed-vga"
	echo "- zed2"
	echo "- zedm"
	echo "Note that the default resolution is HD except for zed-vga"
	echo ""
fi
