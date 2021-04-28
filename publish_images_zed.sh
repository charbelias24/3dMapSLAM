#!/bin/bash

if [ "$#" -ne 2 ] && [ "$#" -ne 1 ]
then
	echo ""
	echo "First argument -- ZED Camera Model"
	echo "- zed"
	echo "- zed2"
	echo "- zedm"

	echo ""
	echo "If you want to publish images from pre-recorded SVO file, add its file path as a second argument"
	echo ""
fi

if [[ "$1" == "zed" ]] 
then
	echo "Launching zed_wrapper with ZED HD camera"
	roslaunch zed_wrapper zed.launch camera_name:=zed  svo_file:=$2
elif [[ "$1" == "zed2" ]] 
then
	echo "Launching zed_wrapper with ZED2 HD camera"
	roslaunch zed_wrapper zed2.launch camera_name:=zed  svo_file:=$2
elif [[ "$1" == "zedm" ]] 
then
	echo "Launching zed_wrapper with ZEDm HD camera"
	roslaunch zed_wrapper zedm.launch camera_name:=zed  svo_file:=$2
fi